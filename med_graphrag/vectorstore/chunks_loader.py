from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple, Dict


def load_chunks_jsonl(path: str | Path) -> List[Dict]:
    """
    Load our preprocessed chunks from a JSONL file.

    Each line is expected to be:
      {"text": "...", "metadata": {...}}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records


def prepare_for_chroma(
    records: Iterable[Dict],
) -> Tuple[List[str], List[str], List[Dict]]:
    """
    From JSONL records → Chroma inputs:
      - ids: unique ID per chunk
      - documents: text
      - metadatas: metadata dicts
    """
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict] = []

    for rec in records:
        text = rec.get("text", "")
        metadata = rec.get("metadata", {})

        # We expect chunk_id in metadata from our splitter; if not, fallback
        chunk_id = metadata.get("chunk_id")
        if not chunk_id:
            # Simple fallback ID
            chunk_id = f"{metadata.get('source', 'unknown')}_p{metadata.get('page', 0)}_idx{metadata.get('chunk_index', 0)}"
            metadata["chunk_id"] = chunk_id

        ids.append(str(chunk_id))
        documents.append(text)
        metadatas.append(metadata)

    return ids, documents, metadatas
