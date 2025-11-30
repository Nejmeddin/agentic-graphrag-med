from __future__ import annotations

from med_graphrag.answering.answerer import answer_question_with_agentic_planner


def main():
    print("🧬 Agentic GraphRAG (Planner) Medical QA Demo")
    print("Ask a medical question based on the textbook.")
    print("Press Enter on an empty line to exit.\n")

    while True:
        question = input("❓ Question: ").strip()
        if not question:
            break

        try:
            answer = answer_question_with_agentic_planner(question)
        except Exception as e:
            print(f"❌ Error while answering: {e}")
            continue

        print("\n📝 Answer:")
        print(answer)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
