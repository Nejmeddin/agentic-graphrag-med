from __future__ import annotations

from med_graphrag.langgraph_app.app import build_qa_app
from med_graphrag.langgraph_app.state import QAState


def main():
    print("🧬 LangGraph Agentic GraphRAG Medical QA Demo")
    print("Ask a medical question based on the textbook.")
    print("Press Enter on an empty line to exit.\n")

    app = build_qa_app()

    while True:
        question = input("❓ Question: ").strip()
        if not question:
            break

        state = QAState(question=question)

        try:
            final_state: QAState = app.invoke(state)
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

        print("\n🧭 Plan:")
        if final_state.get('plan') is not None:
            print(f"  Mode={final_state.get('plan').mode}, n_results={final_state.get('plan').n_results}, "
                  f"hops={final_state.get('plan').neighbor_hops}, use_graph={final_state.get('plan').use_graph}")
            if final_state.get('plan').reason:
                print(f"  Reason: {final_state.get('plan').reason}")

        print("\n📝 Answer:")
        print(final_state.get('answer') or "(no answer)")

        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
