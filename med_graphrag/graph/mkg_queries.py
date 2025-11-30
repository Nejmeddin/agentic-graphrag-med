from __future__ import annotations

from typing import List, Dict, Any

from med_graphrag.graph.neo4j_client import get_driver


def get_chunk_neighbors_and_entities(
    chunk_id: str,
    hops: int = 1,
) -> Dict[str, Any]:
    """
    Récupère un chunk, ses voisins NEXT_CHUNK (avant/après) jusqu'à 'hops',
    et les entités médicales mentionnées par ces chunks.
    """
    driver = get_driver()

    query = """
    MATCH (c:Chunk {chunk_id: $chunk_id})

    // Chunk central + voisins via NEXT_CHUNK (avant et après)
    CALL {
      WITH c
      MATCH path1 = (c)-[:NEXT_CHUNK*0..$hops]->(n1:Chunk)
      OPTIONAL MATCH path2 = (c)<-[:NEXT_CHUNK*0..$hops]-(n2:Chunk)
      WITH COLLECT(DISTINCT n1) + COLLECT(DISTINCT n2) AS neighbors
      UNWIND neighbors AS nb
      WITH DISTINCT nb
      RETURN COLLECT(nb) AS chunks
    }

    // Entités médicales mentionnées par ces chunks
    CALL {
      WITH chunks
      UNWIND chunks AS ch
      OPTIONAL MATCH (ch)-[r:MENTIONS]->(e:MedicalEntity)
      WITH ch, r, e
      WHERE e IS NOT NULL
      RETURN COLLECT(
        DISTINCT {
          chunk_id: ch.chunk_id,
          entity_name: e.name,
          entity_type: e.type,
          confidence: r.confidence
        }
      ) AS entities
    }

    RETURN
      [ch IN chunks |
        {
          chunk_id: ch.chunk_id,
          text: ch.text,
          page_number: ch.page_number,
          source: ch.source,
          chunk_index: ch.chunk_index
        }
      ] AS chunk_context,
      entities AS medical_entities
    """

    with driver.session() as session:
        result = session.run(query, chunk_id=chunk_id, hops=hops)
        record = result.single()
        if not record:
            return {"chunk_context": [], "medical_entities": []}
        return {
            "chunk_context": record["chunk_context"],
            "medical_entities": record["medical_entities"],
        }
