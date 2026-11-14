import json
import re
import statistics
from pathlib import Path

import tiktoken

from src.rag.document_loader import load_policy_documents


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

ENCODING = tiktoken.get_encoding("cl100k_base")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

MAX_TOKENS = 500
TARGET_TOKENS = 350
MIN_TOKENS = 100
OVERLAP_TOKENS = 50


# ---------------------------------------------------------
# TOKEN UTILITIES
# ---------------------------------------------------------

def tokenize(text):
    return ENCODING.encode(text, disallowed_special=())


def count_tokens(text):
    return len(tokenize(text))


def decode_tokens(tokens):
    return ENCODING.decode(tokens)


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text):
    """Normalize extracted PDF text without destroying structure."""

    text = text.replace("\u00a0", " ")
    text = text.replace("\ufb01", "fi")
    text = text.replace("\ufb02", "fl")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------
# HEADING DETECTION
# ---------------------------------------------------------

def is_heading(line):
    """
    Detect structural insurance-policy headings.

    Deliberately avoids treating ordinary uppercase sentences
    or short PDF header/footer lines as clause headings.
    """

    line = line.strip()
    # Recognise insurance endorsement headings (IMT and TA).
    upper_line = line.upper()
    for prefix in ("IMT", "TA"):
        if upper_line.startswith(prefix):
            suffix = upper_line[len(prefix):].lstrip()
            if suffix and suffix[0].isdigit():
                return len(line) <= 180

    if not line or len(line) > 110:
        return False

    words = line.split()

    if len(words) > 15:
        return False

    # Explicit structural headings.
    structural_patterns = [
        r"^(PART|SECTION|CHAPTER)\s+(?:[IVXLCDM]+|\d+)\b",
        r"^(CLAUSE|ANNEXURE|ANNEX|SCHEDULE)\s+[A-Z0-9IVX.-]+\b",
    ]

    for pattern in structural_patterns:
        if re.match(pattern, line, re.IGNORECASE):
            return True

    # Numbered clauses: 1., 1.1, 2.3.1, etc.
    numbered = re.match(
        r"^(\d+(?:\.\d+){0,3})[.)]?\s+(.+)$",
        line,
    )

    if numbered:
        remainder = numbered.group(2)

        if (
            len(remainder.split()) <= 13
            and not re.search(r"[;:]$", remainder)
            and not re.search(r"\b(is|are|means|shall|will)\b", remainder, re.I)
        ):
            return True

    # Roman numeral and lettered headings.
    if re.match(r"^(?:[IVXLCDM]+|[A-Z])[.)]\s+\S", line):
        return len(words) <= 12

    # Common insurance-policy headings.
    known_headings = {
        "definitions",
        "exclusions",
        "scope of cover",
        "geographical area",
        "conditions",
        "general conditions",
        "special conditions",
        "own damage section",
        "third party liability",
        "liability to third parties",
        "indemnity",
        "claims procedure",
        "claim procedure",
        "cancellation",
        "deductibles",
        "limitations as to use",
        "no claim bonus",
        "insured's declared value",
        "personal accident cover",
        "depreciation",
        "add on covers",
        "add-on covers",
    }

    if line.lower().rstrip(":") in known_headings:
        return True

    # Strict uppercase heading detection.
    # Do not apply IGNORECASE to this condition.
    if (
        line == line.upper()
        and any(c.isalpha() for c in line)
        and 2 <= len(words) <= 12
        and not re.search(r"[.;]", line)
    ):
        return True

    return False


# ---------------------------------------------------------
# PARAGRAPH AND SECTION EXTRACTION
# ---------------------------------------------------------

def extract_sections(text):
    """
    Identify headings and group following paragraphs
    into policy sections.
    """

    lines = clean_text(text).splitlines()

    sections = []
    current_heading = "General Policy Text"
    current_content = []

    def save_section():
        content = "\n".join(current_content).strip()

        if content:
            sections.append({
                "heading": current_heading,
                "text": content,
            })

    for line in lines:
        line = line.strip()

        if not line:
            if current_content and current_content[-1] != "":
                current_content.append("")
            continue

        if is_heading(line):
            save_section()

            current_heading = line
            current_content = []
        else:
            current_content.append(line)

    save_section()

    return sections


# ---------------------------------------------------------
# SAFE TEXT SPLITTING
# ---------------------------------------------------------

def split_long_paragraph(text, max_tokens):
    """Split oversized paragraphs at sentence boundaries."""

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    pieces = []
    current = []

    for sentence in sentences:
        if count_tokens(sentence) > max_tokens:
            if current:
                pieces.append(" ".join(current))
                current = []

            tokens = tokenize(sentence)

            for start in range(0, len(tokens), max_tokens):
                part = decode_tokens(tokens[start:start + max_tokens])
                if part.strip():
                    pieces.append(part.strip())

            continue

        candidate = " ".join(current + [sentence])

        if count_tokens(candidate) <= max_tokens:
            current.append(sentence)
        else:
            if current:
                pieces.append(" ".join(current))
            current = [sentence]

    if current:
        pieces.append(" ".join(current))

    return pieces


def split_section(section, max_tokens=MAX_TOKENS):
    """
    Split an oversized section while preserving its heading
    and paragraph boundaries wherever possible.
    """

    heading = section["heading"]
    text = section["text"]

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    heading_tokens = count_tokens(heading)
    content_limit = max_tokens - heading_tokens - 5

    if content_limit < 50:
        content_limit = max_tokens - 5

    pieces = []
    current = []

    for paragraph in paragraphs:
        paragraph_parts = (
            [paragraph]
            if count_tokens(paragraph) <= content_limit
            else split_long_paragraph(paragraph, content_limit)
        )

        for part in paragraph_parts:
            candidate = "\n\n".join(current + [part])

            if count_tokens(candidate) <= content_limit:
                current.append(part)
            else:
                if current:
                    pieces.append("\n\n".join(current))

                current = [part]

    if current:
        pieces.append("\n\n".join(current))

    return [
        {
            "heading": heading,
            "text": piece,
        }
        for piece in pieces
    ]


# ---------------------------------------------------------
# SECTION PACKING
# ---------------------------------------------------------


def pack_sections(sections):
    """Pack sections while isolating General Exceptions boundaries."""
    prepared = []
    for section in sections:
        prepared.extend(split_section(section))

    chunks = []
    current = []
    current_tokens = 0

    for section in prepared:
        rendered = "{}\n{}".format(
            section["heading"], section["text"]
        )
        size = count_tokens(rendered)

        if size > MAX_TOKENS:
            raise ValueError("Section exceeds maximum chunk size.")

        heading = section["heading"].strip().casefold().rstrip(":")
        is_boundary = heading in (
            "general exceptions",
            "general exception",
        )

        if is_boundary:
            if current:
                chunks.append(current)
                current = []
                current_tokens = 0
            chunks.append([rendered])
            continue

        if current and current_tokens + size > TARGET_TOKENS:
            chunks.append(current)
            current = []
            current_tokens = 0

        current.append(rendered)
        current_tokens += size

    if current:
        chunks.append(current)

    return ["\n\n".join(chunk) for chunk in chunks]



# ---------------------------------------------------------
# METADATA AND CHUNK CREATION
# ---------------------------------------------------------

def create_chunk(text, metadata, strategy, index, headings=None):
    text = text.strip()

    chunk_metadata = metadata.copy()

    page = metadata.get("page", "unknown")

    chunk_metadata.update({
        "chunk_id": (
            f"{metadata['file_name']}"
            f"::page-{page}"
            f"::{strategy}"
            f"::{index}"
        ),
        "chunk_index": index,
        "chunking_strategy": strategy,
        "token_count": count_tokens(text),
        "headings": headings or [],
    })

    return {
        "text": text,
        "metadata": chunk_metadata,
    }


# ---------------------------------------------------------
# STRATEGY 1: FIXED SIZE
# ---------------------------------------------------------

def fixed_size_chunks(
    documents,
    chunk_size=MAX_TOKENS,
    overlap=OVERLAP_TOKENS,
):
    results = []

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    for document in documents:
        text = clean_text(document["text"])

        if not text:
            continue

        tokens = tokenize(text)
        step = chunk_size - overlap

        for index, start in enumerate(range(0, len(tokens), step)):
            piece = decode_tokens(tokens[start:start + chunk_size])

            results.append(
                create_chunk(
                    piece,
                    document["metadata"],
                    "fixed_size",
                    index,
                )
            )

            if start + chunk_size >= len(tokens):
                break

    return results


# ---------------------------------------------------------
# STRATEGY 2: HEADING-AWARE CHUNKING
# ---------------------------------------------------------

def classify_policy_scope(text, current_scope="general", current_title="General Policy Text"):
    """Classify policy sections and explicitly numbered endorsements."""
    normalized = re.sub(r"\s+", " ", text).casefold()
    normalized = normalized.replace("–", "-").replace("—", "-")
    if normalized.lstrip().startswith("general exceptions"):
        return "general", "General Exceptions"

    if re.match(r"^\\s*general exceptions?\\b", normalized):
        return "general", "General Exceptions"

    # Explicit main policy sections take precedence over inherited context.
    if "section i - loss of or damage to the vehicle insured" in normalized:
        return "own_damage", "Section I - Loss of or Damage to the Vehicle Insured"
    if "section ii - liability to third parties" in normalized:
        return "third_party", "Section II - Liability to Third Parties"
    if "section iii - personal accident" in normalized:
        return "personal_accident", "Section III - Personal Accident"

    # Recognise endorsements by their own number so one endorsement cannot
    # inherit the subject of a previous endorsement.
    imt_numbers = []
    search_from = 0
    while True:
        position = normalized.find("imt", search_from)
        if position < 0:
            break
        tail = normalized[position + 3:].lstrip()
        digits = ""
        for char in tail:
            if char.isdigit():
                digits += char
            else:
                break
        if digits:
            imt_numbers.append(int(digits))
        search_from = position + 3
    recognised = [n for n in imt_numbers if n in (15, 16, 17, 19, 22, 24)]
    if recognised:
        number = recognised[-1]
        if number in (15, 16, 17):
            return "personal_accident", "Personal Accident Endorsement IMT{}".format(number)
        if number in (19, 22):
            return "own_damage", "Vehicle Damage Endorsement IMT{}".format(number)
        if number == 24:
            return "add_on", "Electrical/Electronic Fittings Add-on IMT24"

    # TA-numbered covers and explicitly named add-on covers are optional benefits.
    if re.search(r"ta\s*\d{1,3}", normalized) or "add-on cover" in normalized or "add on cover" in normalized:
        return "add_on", "Policy Add-on"

    # Reset stale Personal Accident context for unrelated policy topics.
    if "no claim bonus" in normalized:
        return "general", "No Claim Bonus"
    if "conditions during the contract" in normalized or "general conditions" in normalized:
        return "general", "General Policy Conditions"

    # Use supporting injury/compensation language for other PA passages.
    if "personal accident" in normalized and any(
        term in normalized for term in ("bodily injury", "compensation", "death or injury", "passenger")
    ):
        return "personal_accident", "Personal Accident"

    # Unknown IMT endorsements should not inherit an unrelated PA label.
    if imt_numbers:
        return "general", "Policy Endorsement IMT{}".format(imt_numbers[0])

    return current_scope, current_title

def clause_aware_chunks(
    documents,
    target_tokens=TARGET_TOKENS,
    max_tokens=MAX_TOKENS,
    min_tokens=MIN_TOKENS,
):
    results = []
    active_scope_by_file = {}

    for document in documents:
        text = clean_text(document["text"])
        if not text:
            continue

        file_name = document["metadata"]["file_name"]
        scope, title = active_scope_by_file.get(
            file_name, ("general", "General Policy Text")
        )

        sections = extract_sections(text)
        if not sections:
            sections = [{
                "heading": "General Policy Text",
                "text": text,
            }]

        packed = pack_sections(sections)

        if len(packed) >= 2:
            if (
                count_tokens(packed[-1]) < min_tokens
                and count_tokens(packed[-2] + "\n\n" + packed[-1]) <= max_tokens
            ):
                packed[-2] += "\n\n" + packed[-1]
                packed.pop()

        for index, piece in enumerate(packed):
            if count_tokens(piece) > max_tokens:
                raise ValueError(
                    "Chunk exceeds maximum size: {}".format(file_name)
                )

            headings = [
                section["heading"]
                for section in sections
                if section["heading"] in piece
            ]

            scope, title = classify_policy_scope(piece, scope, title)

            chunk = create_chunk(
                piece,
                document["metadata"],
                "clause_aware",
                index,
                headings,
            )
            chunk["metadata"]["policy_scope"] = scope
            chunk["metadata"]["policy_section"] = title
            results.append(chunk)

        active_scope_by_file[file_name] = (scope, title)

    return results

def save_jsonl(chunks, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    return output_path


# ---------------------------------------------------------
# REPORTING
# ---------------------------------------------------------

def print_report(name, chunks):
    print(f"\n{name}")
    print("-" * 45)

    if not chunks:
        print("No chunks generated.")
        return

    sizes = [
        chunk["metadata"]["token_count"]
        for chunk in chunks
    ]

    print("Total chunks:", len(chunks))
    print("Minimum tokens:", min(sizes))
    print("Maximum tokens:", max(sizes))
    print("Average tokens:", round(statistics.mean(sizes), 2))
    print("Median tokens:", statistics.median(sizes))

    print(
        "Chunks below 100 tokens:",
        sum(size < MIN_TOKENS for size in sizes),
    )

    print(
        "Chunks between 100 and 500 tokens:",
        sum(MIN_TOKENS <= size <= MAX_TOKENS for size in sizes),
    )

    print(
        "Chunks above 500 tokens:",
        sum(size > MAX_TOKENS for size in sizes),
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("Loading policy documents...", flush=True)

    documents = load_policy_documents()

    print(f"Loaded {len(documents)} PDF pages.", flush=True)

    print("\nGenerating fixed-size chunks...", flush=True)
    fixed = fixed_size_chunks(documents)

    print("Generating heading-aware chunks...", flush=True)
    clause = clause_aware_chunks(documents)

    print_report("FIXED-SIZE CHUNKING", fixed)
    print_report("HEADING-AWARE CHUNKING", clause)

    fixed_path = save_jsonl(
        fixed,
        "fixed_size_chunks.jsonl",
    )

    clause_path = save_jsonl(
        clause,
        "clause_aware_chunks.jsonl",
    )

    print("\nSAVED OUTPUTS")
    print("-" * 45)
    print(fixed_path)
    print(clause_path)

    if clause:
        print("\nSAMPLE HEADING-AWARE CHUNK")
        print("-" * 45)
        print(clause[0]["text"][:1000])
        print("\nMETADATA")
        print(clause[0]["metadata"])

    print("\nChunking completed successfully.")


if __name__ == "__main__":
    main()
