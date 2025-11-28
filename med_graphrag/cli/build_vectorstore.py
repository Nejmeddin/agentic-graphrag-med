from __future__ import annotations

from pathlib import Path
from typing import List

from med_graphrag.config.settings import settings
from med_graphrag.vectorstore.store_chroma import (
    get_chroma_client,
    get_or_create_collection,
)
from med_graphrag.vectorstore.chunks_loader import (
    load_chunks_jsonl,
    prepare_for_chroma,
)


def batched(iterable, batch_size: int):
    """Simple batching helper."""
    batch: List = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    chunks_path = Path(settings.processed_dir) / "C:\\Users\\User\\Desktop\\agentic-graphrag-med\\data\\processed\\essentials_chunks.jsonl"
    print(f"📂 Loading chunks from {chunks_path} ...")
    records = load_chunks_jsonl(chunks_path)
    print(f"  → Loaded {len(records)} chunk records")

    print("🧱 Preparing data for Chroma...")
    ids, documents, metadatas = prepare_for_chroma(records)
    print(f"  → {len(ids)} IDs / documents / metadatas ready")

    print("🚀 Connecting to Chroma (persistent)...")
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    print(f"  → Using collection: {collection.name}")

    # Safety: you might want to clear the collection if rebuilding
    # collection.delete(ids=ids)  # or collection.reset() to clear all

    print("💾 Inserting chunks into Chroma (batched)...")

    BATCH_SIZE = 128
    total = len(ids)
    for i, batch_indices in enumerate(
        batched(range(total), batch_size=BATCH_SIZE),
        start=1,
    ):
        batch_ids = [ids[j] for j in batch_indices]
        batch_docs = [documents[j] for j in batch_indices]
        batch_metas = [metadatas[j] for j in batch_indices]

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            # embeddings=None → Chroma uses embedding_function automatically
        )
        print(f"  → Batch {i}: added {len(batch_ids)} chunks")

    print("✅ Vector store build complete.")


if __name__ == "__main__":
    main()
