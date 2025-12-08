# med_graphrag/cli/qa_from_image_demo.py
from __future__ import annotations

from med_graphrag.vision.image_classifier import classify_disease_from_image
from med_graphrag.answering.answerer import answer_question_with_agentic_planner


def main():
    print("🧬 Image → Disease → Agentic GraphRAG Demo")
    print("This tool is for educational purposes only.")
    print("It is NOT a medical diagnostic tool.")
    print("Press Enter on an empty line to exit.\n")

    while True:
        image_path = input("Path to disease image").strip()
        if not image_path:
            break

        # 1) Classifier l'image
        try:
            prediction = classify_disease_from_image(image_path)
        except Exception as e:
            print(f"❌ Error while classifying image: {e}")
            continue

        disease_name = prediction.disease_name
        confidence = prediction.confidence

        print(f"\n[ImageClassifier] Predicted disease: {disease_name} (confidence={confidence:.2f})")
        print("⚠️ This is NOT a medical diagnosis. For real health issues, consult a doctor.\n")

        # 2) Construire la question pour le GraphRAG
        # Tu peux adapter cette question à ce que tu veux.
        question_template = (
            "Explain the disease {disease} with its main characteristics: "
            "definition, typical symptoms, possible complications and usual treatments."
        )
        question = question_template.format(disease=disease_name)

        print(f"[GraphRAG] Using auto-generated question:\n  {question}\n")

        # 3) Appeler Agentic GraphRAG (planner + retriever + answerer)
        try:
            answer = answer_question_with_agentic_planner(question)
        except Exception as e:
            print(f"❌ Error while answering question with GraphRAG: {e}")
            continue

        # 4) Afficher la réponse
        print("📝 Answer (educational, not medical advice):\n")
        print(answer)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
