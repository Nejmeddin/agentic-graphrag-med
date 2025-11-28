from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from neo4j import Driver
from med_graphrag.config.settings import settings
from med_graphrag.graph.neo4j_client import get_driver
from med_graphrag.graph.mkg_schema import ensure_constraints
from med_graphrag.vectorstore.chunks_loader import load_chunks_jsonl


DOCUMENT_ID = "essentials_of_human_diseases"
DOCUMENT_TITLE = "Essentials of Human Diseases and Conditions"
DOCUMENT_SOURCE = "essentials-of-human-diseases-and-conditions_compress.pdf"


def _merge_document(tx, doc_id: str, title: str, source: str):
    tx.run(
        """
        MERGE (d:Document {id: $id})
        ON CREATE SET d.title = $title,
                      d.source = $source
        ON MATCH SET d.title = coalesce(d.title, $title),
                     d.source = coalesce(d.source, $source)
        """,
        id=doc_id,
        title=title,
        source=source,
    )


def _merge_page(tx, doc_id: str, page_number: int, source: str):
    page_id = f"{doc_id}_page_{page_number}"
    tx.run(
        """
        MATCH (d:Document {id: $doc_id})
        MERGE (p:Page {id: $page_id})
        ON CREATE SET p.page_number = $page_number,
                      p.source = $source
        MERGE (d)-[:HAS_PAGE]->(p)
        """,
        doc_id=doc_id,
        page_id=page_id,
        page_number=page_number,
        source=source,
    )
    return page_id


def _merge_chunk(
    tx,
    chunk: Dict,
    doc_id: str,
):
    """
    chunk dict format (from JSONL):
      {
        "text": "...",
        "metadata": {
           "chunk_id": "...",
           "page": int,
           "source": "path",
           "chunk_index": int,
           ...
        }
      }
    """
    text: str = chunk.get("text", "")
    metadata: Dict = chunk.get("metadata", {})

    chunk_id: str = metadata.get("chunk_id")
    page_number: int = metadata.get("page", 0)
    source: str = metadata.get("source", DOCUMENT_SOURCE)
    chunk_index: int = metadata.get("chunk_index", 0)

    if not chunk_id:
        # fallback
        chunk_id = f"{doc_id}_p{page_number}_idx{chunk_index}"

    # Ensure page exists & linked to document
    page_id = f"{doc_id}_page_{page_number}"

    tx.run(
        """
        MATCH (d:Document {id: $doc_id})
        MERGE (p:Page {id: $page_id})
          ON CREATE SET p.page_number = $page_number,
                        p.source = $source
        MERGE (d)-[:HAS_PAGE]->(p)

        MERGE (c:Chunk {chunk_id: $chunk_id})
          ON CREATE SET c.text = $text,
                        c.page_number = $page_number,
                        c.source = $source,
                        c.chunk_index = $chunk_index
          ON MATCH SET c.text = $text  // refresh text if changed

        MERGE (p)-[:HAS_CHUNK]->(c)
        """,
        doc_id=doc_id,
        page_id=page_id,
        page_number=page_number,
        source=source,
        chunk_id=chunk_id,
        text=text,
        chunk_index=chunk_index,
    )

    return chunk_id, page_number, chunk_index


def _link_next_chunk(
    tx,
    current_chunk_id: str,
    previous_chunk_id: str | None,
):
    if previous_chunk_id is None:
        return
    tx.run(
        """
        MATCH (c1:Chunk {chunk_id: $prev_id})
        MATCH (c2:Chunk {chunk_id: $cur_id})
        MERGE (c1)-[:NEXT_CHUNK]->(c2)
        """,
        prev_id=previous_chunk_id,
        cur_id=current_chunk_id,
    )


def ingest_chunks_to_neo4j(
    chunks_path: str | Path | None = None,
    driver: Driver | None = None,
):
    """
    Ingest all chunks from JSONL into Neo4j as:
      Document -> Page -> Chunk (+ NEXT_CHUNK chain)
    """
    if driver is None:
        driver = get_driver()

    if chunks_path is None:
        chunks_path = Path(settings.processed_dir) / "essentials_chunks.jsonl"
    chunks_path = Path(chunks_path)

    records: List[Dict] = load_chunks_jsonl(chunks_path)

    ensure_constraints(driver)

    with driver.session() as session:
        # First ensure document node exists
        session.execute_write(
            _merge_document,
            DOCUMENT_ID,
            DOCUMENT_TITLE,
            DOCUMENT_SOURCE,
        )

        previous_chunk_id: str | None = None

        for idx, rec in enumerate(records):
            # MERGE chunk & relationships
            chunk_id, page_number, chunk_index = session.execute_write(
                _merge_chunk,
                rec,
                DOCUMENT_ID,
            )

            # Link sequential chunks to form a path
            session.execute_write(
                _link_next_chunk,
                chunk_id,
                previous_chunk_id,
            )

            previous_chunk_id = chunk_id

            if (idx + 1) % 200 == 0:
                print(f"  → Ingested {idx + 1} chunks...")


def main():
    driver = get_driver()
    try:
        ingest_chunks_to_neo4j(driver=driver)
        print("✅ Basic Medical KG (Document/Page/Chunk) built in Neo4j.")
    finally:
        from med_graphrag.graph.neo4j_client import close_driver
        close_driver()


if __name__ == "__main__":
    main()
