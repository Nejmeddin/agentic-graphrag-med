from __future__ import annotations

from typing import Dict, Any, List

from med_graphrag.graph.neo4j_client import get_driver


def get_chunk_neighbors_and_entities(
    chunk_id: str,
    hops: int = 1,
) -> Dict[str, Any]:
    """
    Récupère :
      - le chunk central
      - ses voisins NEXT_CHUNK (avant/après) jusqu'à 'hops' (si hops > 0)
      - les entités MENTIONS de tous ces chunks

    Version simple, avec:
      - cas spécial pour hops == 0
      - 2 requêtes séparées (chunks, puis entités)
    """
    driver = get_driver()
    if hops < 0:
        hops = 0
    if hops > 3:
        hops = 3  # sécurité

    # --------------------------
    # 1) Récupérer les chunks
    # --------------------------
    with driver.session() as session:
        if hops == 0:
            # 🔹 cas le plus simple: juste le chunk central
            print(f"🔍 [Neo4j] Fetching ONLY central chunk for chunk_id={chunk_id} (hops=0)...")
            query_neighbors = """
            MATCH (c:Chunk {chunk_id: $chunk_id})
            RETURN
              c.chunk_id    AS chunk_id,
              c.text        AS text,
              c.page_number AS page_number,
              c.source      AS source,
              c.chunk_index AS chunk_index
            """
            rows = list(session.run(query_neighbors, chunk_id=chunk_id))
        else:
            # 🔹 cas général: chunk + voisins via NEXT_CHUNK*1..hops
            print(f"🔍 [Neo4j] Fetching chunk neighbors for chunk_id={chunk_id} (hops={hops})...")
            query_neighbors = f"""
            MATCH (c:Chunk {{chunk_id: $chunk_id}})
            OPTIONAL MATCH (c)-[:NEXT_CHUNK*1..{hops}]->(n1:Chunk)
            OPTIONAL MATCH (c)<-[:NEXT_CHUNK*1..{hops}]-(n2:Chunk)
            WITH collect(DISTINCT c) + collect(DISTINCT n1) + collect(DISTINCT n2) AS all_chunks
            UNWIND all_chunks AS ch
            WITH DISTINCT ch
            RETURN
              ch.chunk_id    AS chunk_id,
              ch.text        AS text,
              ch.page_number AS page_number,
              ch.source      AS source,
              ch.chunk_index AS chunk_index
            ORDER BY page_number, chunk_index
            """
            rows = list(session.run(query_neighbors, chunk_id=chunk_id))

        print(f"🔍 [Neo4j] Neighbor query returned {len(rows)} rows.")

        chunk_context: List[Dict[str, Any]] = [
            {
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "page_number": r["page_number"],
                "source": r["source"],
                "chunk_index": r["chunk_index"],
            }
            for r in rows
            if r.get("chunk_id") is not None
        ]

    if not chunk_context:
        print("⚠️ [Neo4j] No chunks found for this chunk_id.")
        return {"chunk_context": [], "medical_entities": []}

    # --------------------------
    # 2) Récupérer les entités
    # --------------------------
    chunk_ids = [c["chunk_id"] for c in chunk_context]

    with driver.session() as session:
        print(f"🔍 [Neo4j] Fetching medical entities for {len(chunk_ids)} chunks...")
        query_entities = """
        MATCH (c:Chunk)-[r:MENTIONS]->(e:MedicalEntity)
        WHERE c.chunk_id IN $chunk_ids
        RETURN
          c.chunk_id   AS chunk_id,
          e.name       AS entity_name,
          e.type       AS entity_type,
          r.confidence AS confidence
        """
        rows_ent = list(session.run(query_entities, chunk_ids=chunk_ids))
        print(f"🔍 [Neo4j] Entities query returned {len(rows_ent)} rows.")

        medical_entities: List[Dict[str, Any]] = [
            {
                "chunk_id": r["chunk_id"],
                "entity_name": r["entity_name"],
                "entity_type": r["entity_type"],
                "confidence": r["confidence"],
            }
            for r in rows_ent
        ]

    return {
        "chunk_context": chunk_context,
        "medical_entities": medical_entities,
    }
