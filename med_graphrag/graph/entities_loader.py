from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict


def load_entities_jsonl(path: str | Path) -> List[Dict]:
    """
    Charge le fichier JSONL produit par extract_medical_entities.py.

    Chaque ligne doit ressembler à :
    {
      "chunk_id": "...",
      "page": 5,
      "source": "...pdf",
      "entities": [
        {
          "name": "...",
          "type": "DISEASE",
          "short_definition": "...",
          "synonyms": ["..."],
          "confidence": 0.9
        },
        ...
      ]
    }
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Entities file not found: {path}")

    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records
