from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from med_graphrag.vectorstore.store_chroma import (
    get_chroma_client,
    get_or_create_collection,
)
from med_graphrag.graph.mkg_queries import get_chunk_neighbors_and_entities


@dataclass
class RetrievedContext:
    query: str
    top_chunks: List[Dict[str, Any]]
    graph_expanded_context: List[Dict[str, Any]]
    medical_entities: List[Dict[str, Any]]


def retrieve_with_vector_and_graph(
    query: str,
    n_results: int = 5,
    neighbor_hops: int = 1,
) -> RetrievedContext:
    """
    1) Interroge Chroma pour obtenir les chunks les plus pertinents pour la requête.
    2) Pour chaque chunk, récupère via Neo4j:
       - le chunk + ses voisins (NEXT_CHUNK)
       - les entités médicales mentionnées
    3) Combine tout dans une structure RetrievedContext.
    """
    # --- 1. Vector search via Chroma ---
    print(f"🔍 Searching for top {n_results} vector hits for query: {query}")
    client = get_chroma_client()
    print("Using Chroma client for vector search...")
    collection = get_or_create_collection(client)
    print(f"Using collection: {collection.name}")

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if "distances" in results else []
    print (f"Found {len(docs)} documents for query '{query}'.")
    top_chunks: List[Dict[str, Any]] = []
    for i, (doc, meta) in enumerate(zip(docs, metadatas), start=0):
        print(f"Processing chunk {i + 1}/{len(docs)}: {meta.get('chunk_id', 'unknown')}")
        top_chunks.append(
            {
                "rank": i + 1,
                "chunk_id": meta.get("chunk_id"),
                "page": meta.get("page"),
                "source": meta.get("source"),
                "chunk_index": meta.get("chunk_index"),
                "text": doc,
                "distance": distances[i] if i < len(distances) else None,
            }
        )
    print(f"Retrieved {len(top_chunks)} top chunks from vector search.")
        

    # --- 2. Graph expansion via Neo4j ---
    graph_chunks_all: List[Dict[str, Any]] = []
    entities_all: List[Dict[str, Any]] = []

    seen_chunk_ids = set()

    for tc in top_chunks:
        print(f"Expanding context for chunk_id={tc['chunk_id']}...")
        cid = tc["chunk_id"]
        if not cid:
            continue

        graph_data = get_chunk_neighbors_and_entities(cid, hops=neighbor_hops)
        chunk_context = graph_data.get("chunk_context", [])
        med_entities = graph_data.get("medical_entities", [])

        # éviter doublons de chunks
        for ch in chunk_context:
            print(f"  Found neighbor chunk_id={ch['chunk_id']} (page {ch['page_number']})")
            if ch["chunk_id"] in seen_chunk_ids:
                continue
            seen_chunk_ids.add(ch["chunk_id"])
            graph_chunks_all.append(ch)

        # entités: on peut les accumuler simplement pour l'instant
        entities_all.extend(med_entities)

    return RetrievedContext(
        query=query,
        top_chunks=top_chunks,
        graph_expanded_context=graph_chunks_all,
        medical_entities=entities_all,
    )
