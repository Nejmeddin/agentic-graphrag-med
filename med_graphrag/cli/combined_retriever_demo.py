from __future__ import annotations

from med_graphrag.retrieval.combined_retriever import retrieve_with_vector_and_graph


def main():
    print("🔍 Combined Vector + Graph retriever demo.")
    print("Ask a medical question (press Enter on empty line to exit).")

    while True:
        query = input("\n❓ Query: ").strip()
        if not query:
            break

        ctx = retrieve_with_vector_and_graph(
            query=query,
            n_results=3,
            neighbor_hops=1,
        )

        print("\n=== Top vector hits (Chroma) ===")
        for tc in ctx.top_chunks:
            print(f"\n[{tc['rank']}] chunk_id={tc['chunk_id']} page={tc['page']} distance={tc['distance']}")
            print(tc["text"][:400] + ("..." if len(tc["text"]) > 400 else ""))

        print("\n=== Graph-expanded context (neighbors) ===")
        # On affiche juste quelques chunks de contexte
        for i, ch in enumerate(ctx.graph_expanded_context[:5], start=1):
            print(f"\n[Neighbor {i}] chunk_id={ch['chunk_id']} page={ch['page_number']}")
            print(ch["text"][:300] + ("..." if len(ch["text"]) > 300 else ""))

        print("\n=== Medical entities (from Graph) ===")
        # On peut faire un petit résumé par type
        if not ctx.medical_entities:
            print("No entities found in this context.")
        else:
            for ent in ctx.medical_entities[:10]:
                print(
                    f"- {ent['entity_type']} :: {ent['entity_name']} "
                    f"(from chunk {ent['chunk_id']}, conf={ent.get('confidence')})"
                )


if __name__ == "__main__":
    main()
