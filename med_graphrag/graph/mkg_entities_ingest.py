from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from neo4j import Driver

from med_graphrag.config.settings import settings
from med_graphrag.graph.neo4j_client import get_driver
from med_graphrag.graph.mkg_schema import ensure_constraints
from med_graphrag.graph.entities_loader import load_entities_jsonl


def _merge_medical_entity_and_link(
    tx,
    chunk_id: str,
    entity: Dict,
    confidence_threshold: float = 0.5,
):
    """
    Crée/MET à jour un noeud MedicalEntity et une relation MENTIONS depuis le Chunk.

    entity dict:
      {
        "name": "...",
        "type": "DISEASE" | "SYMPTOM" | "TREATMENT" | "TEST" | "OTHER",
        "short_definition": "...",
        "synonyms": [...],
        "confidence": 0.9
      }
    """
    name: str = (entity.get("name") or "").strip()
    ent_type: str = entity.get("type") or "OTHER"
    short_def: str | None = entity.get("short_definition")
    synonyms: List[str] = entity.get("synonyms") or []
    confidence: float = float(entity.get("confidence", 0.7))

    # Filtrage de base
    if not name:
        return
    if confidence < confidence_threshold:
        return

    # On ignore OTHER pour le moment (optionnel)
    if ent_type == "OTHER":
        return

    tx.run(
        """
        // Assurer que le Chunk existe
        MERGE (c:Chunk {chunk_id: $chunk_id})

        // Créer ou mettre à jour l'entité médicale
        MERGE (e:MedicalEntity {type: $type, name: $name})
          ON CREATE SET
            e.short_definition = $short_definition,
            e.synonyms = $synonyms
          ON MATCH SET
            e.short_definition = coalesce(e.short_definition, $short_definition)

        // Créer la relation MENTIONS (avec confiance)
        MERGE (c)-[r:MENTIONS]->(e)
          ON CREATE SET r.confidence = $confidence
          ON MATCH SET r.confidence = coalesce(r.confidence, $confidence)
        """,
        chunk_id=chunk_id,
        type=ent_type,
        name=name,
        short_definition=short_def,
        synonyms=synonyms,
        confidence=confidence,
    )


def ingest_entities_to_neo4j(
    entities_path: str | Path | None = None,
    driver: Driver | None = None,
    max_records: int | None = None,
):
    """
    Lit entities_essentials.jsonl et injecte:
      - des noeuds :MedicalEntity
      - des relations (Chunk)-[:MENTIONS]->(MedicalEntity)
    """
    if driver is None:
        driver = get_driver()

    if entities_path is None:
        entities_path = Path(settings.processed_dir) / "entities_essentials.jsonl"
    entities_path = Path(entities_path)

    print(f"📂 Loading entities from {entities_path} ...")
    records = load_entities_jsonl(entities_path)
    print(f"  → Loaded {len(records)} entity records")

    if max_records is not None:
        records = records[:max_records]
        print(f"  → Using only first {len(records)} records (max_records)")

    ensure_constraints(driver)

    with driver.session() as session:
        total_links = 0
        total_entities = 0

        for idx, rec in enumerate(records, start=1):
            chunk_id: str = rec.get("chunk_id")
            entities: List[Dict] = rec.get("entities", [])

            for ent in entities:
                session.execute_write(
                    _merge_medical_entity_and_link,
                    chunk_id,
                    ent,
                )
                total_entities += 1
                total_links += 1

            if idx % 20 == 0:
                print(f"  → Processed {idx} chunk-entity groups")

    print(f"✅ Ingestion done: {total_entities} entities linked from {len(records)} chunks.")


def main():
    driver = get_driver()
    try:
        ingest_entities_to_neo4j(driver=driver)
    finally:
        from med_graphrag.graph.neo4j_client import close_driver
        close_driver()


if __name__ == "__main__":
    main()
