import csv
import re
from pathlib import Path
from collections import Counter

import pdfplumber
from pypdf import PdfReader


ROOT_DIR = Path(__file__).resolve().parents[2]
POLICY_DIR = ROOT_DIR / "data" / "policy_docs"
SOURCES_CSV = POLICY_DIR / "sources.csv"


def clean_text(text):
    """Normalize extracted PDF text without changing its wording."""
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\u00a0", " ")
    text = text.replace("\ufeff", "")

    # Remove trailing spaces while preserving line breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_standard_pdf(pdf_path, base_metadata):
    """Extract pages using pypdf for the other insurers."""
    reader = PdfReader(str(pdf_path))
    records = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")

        metadata = {
            **base_metadata,
            "page": page_number,
            "extraction_method": "pypdf",
        }

        records.append({
            "text": text,
            "metadata": metadata,
        })

    return records



def extract_sbi_page(page, page_number):
    """
    Extract SBI General pages column by column, preserving
    the complete page height to avoid clipping policy text.
    """
    width = float(page.width)
    height = float(page.height)

    column_count = 3 if 13 <= page_number <= 18 else 2
    extracted_columns = []
    city_header_seen = False

    footer_markers = [
        "Corporate & Registered Office:",
        "U66000MH2009PLC190546",
        "SBI Logo displayed belongs",
        "customer.care@sbigeneral.in",
        "www.sbigeneral.in",
    ]

    for column_index in range(column_count):
        x0 = width * column_index / column_count
        x1 = width * (column_index + 1) / column_count

        # Keep a small gutter between adjacent columns.
        if column_index > 0:
            x0 += 8
        if column_index < column_count - 1:
            x1 -= 8

        column = page.crop((x0, 0, x1, height))

        raw_text = column.extract_text(
            x_tolerance=2,
            y_tolerance=3
        ) or ""

        kept_lines = []

        for line in raw_text.splitlines():
            cleaned_line = line.strip()

            if not cleaned_line:
                continue

            if cleaned_line == "SBI General Insurance Company Limited":
                continue

            if any(
                marker.lower() in cleaned_line.lower()
                for marker in footer_markers
            ):
                continue

            # The city table repeats its column heading.
            # Keep it only once in the combined page text.
            if cleaned_line == "City Name State":
                if city_header_seen:
                    continue
                city_header_seen = True

            kept_lines.append(cleaned_line)

        if kept_lines:
            extracted_columns.append("\n".join(kept_lines))

    return clean_text("\n\n".join(extracted_columns))

def extract_sbi_pdf(pdf_path, base_metadata):
    """Extract SBI General with layout-aware column handling."""
    records = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = extract_sbi_page(page, page_number)

            if 13 <= page_number <= 18:
                method = "pdfplumber_three_column"
            else:
                method = "pdfplumber_two_column"

            metadata = {
                **base_metadata,
                "page": page_number,
                "extraction_method": method,
            }

            records.append({
                "text": text,
                "metadata": metadata,
            })

    return records


def load_policy_documents():
    """Load policy metadata and extract text page by page."""
    if not SOURCES_CSV.exists():
        raise FileNotFoundError(
            f"Policy source manifest not found: {SOURCES_CSV}"
        )

    all_pages = []

    with SOURCES_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as csv_file:
        sources = list(csv.DictReader(csv_file))

    for source in sources:
        file_name = (source.get("file_name") or "").strip()
        insurer = (source.get("insurer") or "").strip()

        if not file_name:
            raise ValueError("A policy source record has no file_name.")

        pdf_path = POLICY_DIR / file_name

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"Policy PDF not found: {pdf_path}"
            )

        base_metadata = {
            "file_name": file_name,
            "insurer": insurer,
            "document_name": (
                source.get("document_name") or ""
            ).strip(),
            "product_type": (
                source.get("product_type") or ""
            ).strip(),
            "uin": (source.get("uin") or "").strip(),
            "source_url": (
                source.get("source_url") or ""
            ).strip(),
        }

        if insurer.lower() == "sbi general":
            pages = extract_sbi_pdf(pdf_path, base_metadata)
        else:
            pages = extract_standard_pdf(pdf_path, base_metadata)

        all_pages.extend(pages)

    return all_pages


def print_report(pages):
    """Print extraction statistics."""
    insurer_counts = Counter(
        page["metadata"]["insurer"] for page in pages
    )

    method_counts = Counter(
        page["metadata"]["extraction_method"] for page in pages
    )

    empty_pages = sum(
        not page["text"].strip() for page in pages
    )

    print("\nPOLICY DOCUMENT LOADER REPORT")
    print("=" * 45)
    print("Policy PDFs:", len(set(
        p["metadata"]["file_name"] for p in pages
    )))
    print("Total PDF pages:", len(pages))
    print("Pages with extracted text:", len(pages) - empty_pages)
    print("Pages with no extracted text:", empty_pages)

    print("\nInsurers:")
    for insurer, count in sorted(insurer_counts.items()):
        print(f"- {insurer}: {count} pages")

    print("\nExtraction methods:")
    for method, count in sorted(method_counts.items()):
        print(f"- {method}: {count} pages")


if __name__ == "__main__":
    documents = load_policy_documents()
    print_report(documents)
    print("\nDocument loading completed.")
