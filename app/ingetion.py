import os
import re
from typing import List, Dict
from datetime import datetime
from pypdf import PdfReader

from config import CHUNK_SIZE, CHUNK_OVERLAP


# ── Text Extraction ──────────────────────────────────────────────────────────

def extract_pages_from_pdf(file_path: str) -> List[Dict]:
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


def extract_pages_from_txt(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [{"page": 1, "text": f.read()}]


def extract_pages(file_path: str) -> List[Dict]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_pages_from_pdf(file_path)
    elif ext in (".txt", ".md"):
        return extract_pages_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── Chunking ─────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:

    full_text = ""
    char_to_page: List[int] = []

    for p in pages:
        cleaned = clean_text(p["text"])
        if cleaned:
            if full_text:
                full_text += " "
                char_to_page.append(p["page"])
            full_text += cleaned
            char_to_page.extend([p["page"]] * len(cleaned))

    chunks: List[Dict] = []
    start = 0

    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunk = full_text[start:end].strip()

        if chunk:
            page_set = sorted(set(char_to_page[start:end]))
            chunks.append({"chunk_text": chunk, "pages": page_set})

        start += chunk_size - overlap

    return chunks


# ── Versioned Ingestion ──────────────────────────────────────────────────────

def ingest_document(
    file_path: str,
    doc_id: str,
    version: int,
    is_active: bool = True
) -> List[Dict]:
    """
    Version-aware ingestion pipeline.

    Returns records ready for vector DB upsert.
    """

    file_name = os.path.basename(file_path)
    pages = extract_pages(file_path)
    chunks = chunk_pages(pages)

    ingestion_time = datetime.utcnow().isoformat()

    records = []

    for idx, chunk in enumerate(chunks):
        page_str = ",".join(str(p) for p in chunk["pages"])

        records.append(
            {
                # 👇 Version-safe ID
                "id": f"{doc_id}::v{version}::chunk-{idx}",

                # Text for embedding
                "chunk_text": chunk["chunk_text"],

                # Metadata (VERY IMPORTANT)
                "metadata": {
                    "doc_id": doc_id,
                    "doc_version": version,
                    "is_active": is_active,
                    "source": file_name,
                    "pages": page_str,
                    "ingested_at": ingestion_time,
                },
            }
        )

    print(
        f"Ingested '{file_name}' as doc_id={doc_id}, version={version} "
        f"→ {len(records)} chunks"
    )

    return records

if __name__ == "__main__":
    import sys

    print("=== Ingestion Test (Versioned) ===")

    test_path = sys.argv[1] if len(sys.argv) > 1 else "docs/Apple_Q24.pdf"

    # 👇 Logical document identity (NOT filename)
    doc_id = "apple_q2_report"

    # 👇 Manually set version for testing
    version = 1

    print(f"Testing with: {test_path}")
    print(f"Doc ID: {doc_id}")
    print(f"Version: {version}")

    records = ingest_document(
        file_path=test_path,
        doc_id=doc_id,
        version=version,
        is_active=True
    )

    print(f"\nTotal chunks: {len(records)}")

    print("\nFirst chunk preview:")
    print(f"  ID        : {records[0]['id']}")
    print(f"  Doc ID    : {records[0]['metadata']['doc_id']}")
    print(f"  Version   : {records[0]['metadata']['doc_version']}")
    print(f"  Active    : {records[0]['metadata']['is_active']}")
    print(f"  Source    : {records[0]['metadata']['source']}")
    print(f"  Pages     : {records[0]['metadata']['pages']}")
    print(f"  Text      : {records[0]['chunk_text'][:200]}...")

    print("\nLast chunk preview:")
    print(f"  ID        : {records[-1]['id']}")
    print(f"  Text      : {records[-1]['chunk_text'][:200]}...")

    print("✅ Versioned ingestion test passed!")





