"""
Hybrid insurance-policy retriever.

Components:
- BGE-small semantic embeddings
- Persistent ChromaDB vector index
- BM25 lexical retrieval
- Reciprocal Rank Fusion (RRF)
- Insurer and UIN metadata filtering
- Page-level evidence provenance

Python 3.9 compatible.
"""

import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import defaultdict

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parents[2]

CHUNK_PATH = ROOT_DIR / "data" / "processed" / "clause_aware_chunks.jsonl"
VECTOR_DB_PATH = ROOT_DIR / "data" / "vector_db"

COLLECTION_NAME = "vehicle_insurance_policies"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

BATCH_SIZE = 32
DEFAULT_TOP_K = 5
BM25_CANDIDATES = 15
VECTOR_CANDIDATES = 15
RRF_K = 60


def tokenize(text: str) -> List[str]:
    """Tokenize text for BM25."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


# Insurance terminology expansion.
# These are search synonyms, not policy conclusions.
QUERY_EXPANSIONS = {
    "alcohol": "intoxicating liquor",
    "drunk": "intoxicating liquor",
    "drunken": "intoxicating liquor",
    "driving under influence": "under the influence",
    "under influence": "under the influence",
    "vehicle damage": "accidental loss or damage to the vehicle insured",
    "car damage": "accidental loss or damage to the vehicle insured",
    "owner permission": "knowledge and consent of the Insured",
    "owner consent": "knowledge and consent of the Insured",
    "driving licence": "effective driving license valid driving license",
    "license": "licence",
    "theft": "burglary housebreaking theft",
    "flood": "flood inundation storm tempest cyclone",
    "depreciation": "depreciation deduction replaced parts age of vehicle",
    "wear and tear": "wear and tear mechanical or electrical breakdown",
}


def expand_insurance_query(query):
    """
    Add policy-language synonyms to a natural-language query.

    Original wording is retained. Expansion terms only improve
    retrieval recall and must never be treated as evidence.
    """
    normalized = query.lower()
    additions = []

    for phrase, expansion in QUERY_EXPANSIONS.items():
        if phrase in normalized:
            additions.append(expansion)

    if not additions:
        return query

    return query + " " + " ".join(additions)



def query_policy_scope(query: str) -> Optional[str]:
    """Infer the policy section most relevant to a retrieval query."""
    text = " ".join(query.casefold().replace("-", " ").split())

    if any(term in text for term in (
        "third party", "third parties", "third party liability",
        "liability to third parties", "third party property",
    )):
        return "third_party"

    if any(term in text for term in (
        "personal accident", "owner driver", "unnamed passenger",
        "paid driver", "bodily injury", "passenger compensation",
    )):
        return "personal_accident"

    if any(term in text for term in (
        "add on", "engine protect", "zero depreciation",
        "return to invoice", "roadside assistance", "consumables",
        "key replacement", "electrical electronic fittings",
        "personal belongings", "tyre protect", "rim protect",
    )):
        return "add_on"

    if any(term in text for term in (
        "own damage", "vehicle is damaged", "vehicle damage",
        "damage to the vehicle", "loss or damage to the vehicle",
        "accidental loss", "depreciation", "replaced parts",
        "wear and tear", "intoxicating liquor", "under the influence",
        "under influence", "alcohol", "drugs", "drunk",
        "flood", "theft", "fire",
    )):
        return "own_damage"

    return None


class HybridPolicyRetriever:
    """Retrieve policy clauses using vector and keyword search."""

    def __init__(
        self,
        chunk_path: Path = CHUNK_PATH,
        db_path: Path = VECTOR_DB_PATH,
        model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.chunk_path = Path(chunk_path)
        self.db_path = Path(db_path)
        self.model_name = model_name

        self.model = SentenceTransformer(model_name)

        self.client = chromadb.PersistentClient(
            path=str(self.db_path)
        )

        self.collection = None

        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.ids: List[str] = []

        self.bm25 = None
        self.bm25_tokens: List[List[str]] = []

        self.initialized = False

    def load_chunks(self) -> List[Dict[str, Any]]:
        """Read and validate the generated chunk JSONL file."""
        if not self.chunk_path.exists():
            raise FileNotFoundError(
                "Chunk file not found: {}".format(self.chunk_path)
            )

        chunks = []

        with self.chunk_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Invalid JSON on line {}: {}".format(
                            line_number, exc
                        )
                    )

                text = chunk.get("text", "").strip()
                metadata = chunk.get("metadata", {})

                if not text:
                    continue

                chunk_id = metadata.get("chunk_id")

                if not chunk_id:
                    raise ValueError(
                        "Chunk missing chunk_id on line {}".format(
                            line_number
                        )
                    )

                required = ["insurer", "file_name", "page"]

                missing = [
                    key for key in required
                    if metadata.get(key) is None
                ]

                if missing:
                    raise ValueError(
                        "Chunk {} missing metadata: {}".format(
                            chunk_id, missing
                        )
                    )

                chunks.append(chunk)

        if not chunks:
            raise ValueError("No usable chunks found.")

        ids = [
            str(c["metadata"]["chunk_id"]) for c in chunks
        ]

        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate chunk IDs found.")

        return chunks

    @staticmethod
    def chroma_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metadata to Chroma-compatible scalar values."""
        output = {}

        for key, value in metadata.items():
            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                output[key] = value

            elif isinstance(value, list):
                output[key] = ", ".join(
                    str(item) for item in value
                )

            else:
                output[key] = str(value)

        return output

    def rebuild_index(self) -> int:
        """
        Rebuild the vector index from the canonical JSONL chunks.

        This intentionally replaces the existing collection.
        """
        chunks = self.load_chunks()

        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        self.documents = [
            c["text"] for c in chunks
        ]

        self.metadatas = [
            self.chroma_metadata(c["metadata"])
            for c in chunks
        ]

        self.ids = [
            str(c["metadata"]["chunk_id"])
            for c in chunks
        ]

        print("Building BGE embeddings...")

        for start in range(0, len(chunks), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(chunks))

            batch_texts = self.documents[start:end]

            embeddings = self.model.encode(
                batch_texts,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            self.collection.add(
                ids=self.ids[start:end],
                documents=batch_texts,
                metadatas=self.metadatas[start:end],
                embeddings=embeddings.tolist(),
            )

            print(
                "Indexed {}/{} chunks".format(end, len(chunks))
            )

        self.build_bm25()

        self.initialized = True

        return len(chunks)

    def build_bm25(self) -> None:
        """Build the lexical search index."""
        self.bm25_tokens = [
            tokenize(document)
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(self.bm25_tokens)

    def load_existing_index(self) -> bool:
        """
        Load an existing Chroma collection and rebuild BM25
        from the canonical chunk file.
        """
        try:
            self.collection = self.client.get_collection(
                COLLECTION_NAME
            )
        except Exception:
            return False

        chunks = self.load_chunks()

        self.documents = [
            c["text"] for c in chunks
        ]

        self.metadatas = [
            self.chroma_metadata(c["metadata"])
            for c in chunks
        ]

        self.ids = [
            str(c["metadata"]["chunk_id"])
            for c in chunks
        ]

        stored_count = self.collection.count()

        if stored_count != len(self.ids):
            print(
                "Vector index has {} records, but JSONL has {}. "
                "Rebuild required.".format(
                    stored_count, len(self.ids)
                )
            )
            return False

        self.build_bm25()
        self.initialized = True

        return True

    def initialize(self, rebuild: bool = False) -> None:
        """Initialize vector and BM25 indexes."""
        if rebuild:
            count = self.rebuild_index()
            print("Index rebuilt: {} chunks.".format(count))
            return

        if self.load_existing_index():
            print(
                "Existing index loaded: {} chunks.".format(
                    len(self.documents)
                )
            )
        else:
            count = self.rebuild_index()
            print("Index created: {} chunks.".format(count))

    @staticmethod
    def metadata_matches(
        metadata: Dict[str, Any],
        insurer: Optional[str] = None,
        uin: Optional[str] = None,
    ) -> bool:
        """Apply exact insurer and UIN filters."""
        if insurer:
            if str(metadata.get("insurer", "")).casefold() != insurer.casefold():
                return False

        if uin:
            if str(metadata.get("uin", "")).casefold() != uin.casefold():
                return False

        return True

    def retrieve(
        self,
        query: str,
        insurer: Optional[str] = None,
        uin: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict[str, Any]]:
        """Run BM25 + vector search followed by RRF."""
        if not self.initialized:
            raise RuntimeError(
                "Retriever is not initialized. Call initialize() first."
            )

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        # Fail closed: never silently mix insurers.
        if insurer or uin:
            available = [
                metadata
                for metadata in self.metadatas
                if self.metadata_matches(
                    metadata, insurer=insurer, uin=uin
                )
            ]

            if not available:
                raise ValueError(
                    "No policy documents match the requested "
                    "insurer/UIN filter."
                )

        eligible_indices = [
            i for i, metadata in enumerate(self.metadatas)
            if self.metadata_matches(
                metadata, insurer=insurer, uin=uin
            )
        ]

        if not eligible_indices:
            return []

        expanded_query = expand_insurance_query(query)
        query_tokens = tokenize(expanded_query)

        bm25_scores = self.bm25.get_scores(query_tokens)

        bm25_ranked = sorted(
            eligible_indices,
            key=lambda i: float(bm25_scores[i]),
            reverse=True,
        )[:BM25_CANDIDATES]

        bm25_rank = {
            index: rank
            for rank, index in enumerate(bm25_ranked, start=1)
        }

        query_embedding = self.model.encode(
            ["Represent this sentence for searching relevant passages: " + expanded_query],
            normalize_embeddings=True,
        )[0]

        where = None

        conditions = []

        if insurer:
            conditions.append({"insurer": insurer})

        if uin:
            conditions.append({"uin": uin})

        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        vector_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(
                VECTOR_CANDIDATES, len(eligible_indices)
            ),
            where=where,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        vector_ids = vector_results.get("ids", [[]])[0]

        vector_rank = {
            chunk_id: rank
            for rank, chunk_id in enumerate(
                vector_ids, start=1
            )
        }

        index_by_id = {
            chunk_id: i
            for i, chunk_id in enumerate(self.ids)
        }

        candidate_ids = set(
            self.ids[i] for i in bm25_ranked
        )
        candidate_ids.update(vector_ids)

        desired_scope = query_policy_scope(query)
        fused = []

        for chunk_id in candidate_ids:
            index = index_by_id[chunk_id]
            score = 0.0

            if index in bm25_rank:
                score += 1.0 / (RRF_K + bm25_rank[index])

            if chunk_id in vector_rank:
                score += 1.0 / (RRF_K + vector_rank[chunk_id])

            actual_scope = self.metadatas[index].get(
                "policy_scope", "general"
            )
            scope_match = (
                desired_scope is not None
                and actual_scope == desired_scope
            )
            scope_bonus = 0.01 if scope_match else 0.0
            clause_bonus = 0.0

            normalized_text = " ".join(
                self.documents[index].casefold().split()
            )
            query_mentions_intoxication = any(
                term in query.casefold()
                for term in ("alcohol", "drugs", "drunk", "intoxicating liquor")
            )

            if desired_scope == "own_damage" and query_mentions_intoxication:
                has_intoxication_wording = (
                    "intoxicating liquor" in normalized_text
                    and "under the influence" in normalized_text
                )

                has_vehicle_damage_exclusion = (
                    any(phrase in normalized_text for phrase in (
                        "accidental loss or damage suffered whilst",
                        "accidental loss or damage suffered while",
                        "accidental loss or damage whilst",
                        "accidental loss or damage while",
                    ))
                    and "knowledge and consent" in normalized_text
                )

                has_personal_accident_wording = any(
                    phrase in normalized_text
                    for phrase in (
                        "bodily injury",
                        "death or injury",
                        "scale of compensation",
                        "compensation shall be payable",
                        "personal accident",
                        "paid driver",
                        "unnamed passengers",
                    )
                )

                if has_intoxication_wording and has_vehicle_damage_exclusion:
                    clause_bonus = 0.04
                elif has_intoxication_wording and has_personal_accident_wording:
                    clause_bonus = -0.025

            ranking_score = score + scope_bonus + clause_bonus

            fused.append((
                chunk_id, score, ranking_score, index, scope_match
            ))

        # Prefer the intended policy section, then hybrid relevance.
        # Chunk ID provides deterministic tie-breaking.
        fused.sort(
            key=lambda item: (
                -item[2], -item[1], item[0]
            )
        )

        results = []

        for chunk_id, score, ranking_score, index, scope_match in fused[:top_k]:
            metadata = dict(self.metadatas[index])

            results.append({
                "chunk_id": chunk_id,
                "text": self.documents[index],
                "metadata": metadata,
                "hybrid_score": round(score, 6),
                "ranking_score": round(ranking_score, 6),
                "scope_match": scope_match,
                "matched_policy_scope": metadata.get("policy_scope"),
                "citation": {
                    "insurer": metadata.get("insurer"),
                    "document": metadata.get("document_name"),
                    "file_name": metadata.get("file_name"),
                    "page": metadata.get("page"),
                    "source_url": metadata.get("source_url"),
                    "uin": metadata.get("uin"),
                },
            })

        return results


def main():
    """Build the index and run a retrieval smoke test."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid insurance policy retriever"
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the persistent vector index",
    )

    parser.add_argument(
        "--query",
        type=str,
        default="What depreciation applies to replaced vehicle parts?",
    )

    parser.add_argument(
        "--insurer",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--uin",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    retriever = HybridPolicyRetriever()
    retriever.initialize(rebuild=args.rebuild)

    results = retriever.retrieve(
        query=args.query,
        insurer=args.insurer,
        uin=args.uin,
        top_k=args.top_k,
    )

    print("\nRETRIEVAL RESULTS")
    print("=" * 65)

    for rank, result in enumerate(results, start=1):
        citation = result["citation"]

        print(
            "\n{}. {} | Page {} | Score {}".format(
                rank,
                citation["insurer"],
                citation["page"],
                result["hybrid_score"],
            )
        )

        print("Document:", citation["document"])
        print("UIN:", citation["uin"])
        print("Chunk ID:", result["chunk_id"])
        print("Text:", result["text"][:700])


if __name__ == "__main__":
    main()
