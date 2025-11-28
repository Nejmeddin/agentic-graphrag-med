from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document  # type: ignore

from med_graphrag.config.settings import settings


def build_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> RecursiveCharacterTextSplitter:
    """
    Create a RecursiveCharacterTextSplitter with good defaults.

    Best-practice defaults for RAG are usually ~500–1000 chars chunk size
    and 50–200 overlap, depending on docs and model limits. :contentReference[oaicite:5]{index=5}
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )


def generate_chunk_id(doc: Document, chunk_index: int) -> str:
    """
    Generate a stable unique ID for each chunk (idea inspired by similar
    patterns in many RAG blogs). :contentReference[oaicite:6]{index=6}
    """
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", 0)

    content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:8]
    return f"{Path(source).stem}_page{page}_chunk{chunk_index}_{content_hash}"


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.
    Adds useful metadata for each chunk.
    """
    text_splitter = build_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    print("✂️  Splitting documents into chunks...")
    all_chunks: List[Document] = []

    for doc in documents:
        # split_documents expects a list of Document objects
        chunks = text_splitter.split_documents([doc])

        for i, chunk in enumerate(chunks):
            # enrich metadata
            chunk.metadata.setdefault("source", doc.metadata.get("source"))
            chunk.metadata.setdefault("page", doc.metadata.get("page"))

            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks_in_doc"] = len(chunks)
            chunk.metadata["chunk_id"] = generate_chunk_id(chunk, i)

            all_chunks.append(chunk)

    print(f"  → Created {len(all_chunks)} chunks")
    return all_chunks


def save_chunks_jsonl(chunks: List[Document], output_path: Path | None = None) -> Path:
    """
    Save chunks as JSONL with fields:
      - text
      - metadata (includes page, source, chunk_id, etc.)
    """
    if output_path is None:
        output_dir = Path(settings.processed_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "essentials_chunks.jsonl"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            record = {
                "text": ch.page_content,
                "metadata": ch.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"💾 Chunks saved to: {output_path}")
    return output_path
