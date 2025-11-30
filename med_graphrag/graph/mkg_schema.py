from neo4j import Driver
from med_graphrag.graph.neo4j_client import get_driver


def ensure_constraints(driver: Driver | None = None):
    """
    Create uniqueness constraints for our basic Medical KG:
      - Document.id
      - Page.id
      - Chunk.chunk_id
    """
    if driver is None:
        driver = get_driver()

    cypher_statements = [
        "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS "
        "FOR (d:Document) REQUIRE d.id IS UNIQUE",

        "CREATE CONSTRAINT page_id_unique IF NOT EXISTS "
        "FOR (p:Page) REQUIRE p.id IS UNIQUE",

        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
        "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            # 🔹 NEW: MedicalEntity unique by (type, name)
        "CREATE CONSTRAINT med_entity_unique IF NOT EXISTS "
        "FOR (e:MedicalEntity) REQUIRE (e.type, e.name) IS UNIQUE",
    ]

    with driver.session() as session:
        for stmt in cypher_statements:
            session.run(stmt)
