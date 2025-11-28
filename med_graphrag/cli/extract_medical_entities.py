from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from med_graphrag.config.settings import settings
from med_graphrag.vectorstore.chunks_loader import load_chunks_jsonl
from med_graphrag.agents.extraction_agent import build_extraction_chain


def batched(iterable, batch_size: int):
    batch: List = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    chunks_path = Path(settings.processed_dir) / "essentials_chunks.jsonl"
    output_path = Path(settings.processed_dir) / "entities_essentials.jsonl"

    print(f"📂 Loading chunks from {chunks_path} ...")
    records = load_chunks_jsonl(chunks_path)
    print(f"  → Loaded {len(records)} chunk records")

    chain = build_extraction_chain()

    # ⚠️ Pour commencer, limitons le nombre de chunks pour éviter les coûts
    # Tu pourras augmenter ce nombre plus tard.
    MAX_CHUNKS = 50  # par ex. tester sur 50 chunks
    records = records[:MAX_CHUNKS]

    print(f"🧠 Running LLM extraction on {len(records)} chunks...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f_out:
        for idx, rec in enumerate(records, start=1):
            text = rec.get("text", "")
            meta: Dict[str, Any] = rec.get("metadata", {})

            chunk_id = meta.get("chunk_id", f"chunk_{idx}")
            page = meta.get("page", 0)
            source = meta.get("source", settings.pdf_path)

            # Input pour la chaîne
            llm_input = {
                "chunk_text": text,
                "chunk_id": chunk_id,
                "page": page,
                "source": source,
            }

            try:
                result = chain.invoke(llm_input)
            except Exception as e:
                print(f"  ❌ Error extracting chunk {chunk_id}: {e}")
                continue

            # result est un ChunkExtractionResult (Pydantic)
            record_out = result.model_dump()

            f_out.write(json.dumps(record_out, ensure_ascii=False) + "\n")

            if idx % 10 == 0:
                print(f"  → Processed {idx} chunks")

    print(f"✅ Extraction done. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
