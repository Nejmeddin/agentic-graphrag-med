# med_graphrag/vision/image_classifier.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class DiseasePrediction:
    disease_name: str
    confidence: float


class ImageDiseaseClassifier:
    """
    Simple wrapper for an image disease classifier.

    IMPORTANT:
    - This is for educational purposes only.
    - It is NOT a medical diagnostic tool.
    """

    def __init__(self):
        # TODO: load your real model here (PyTorch, etc.)
        # For now we just simulate a few classes.
        self.known_diseases: List[str] = [
            "cystic fibrosis",
            "psoriasis",
            "eczema",
            "acne vulgaris",
            "melanoma",
        ]

    def predict(self, image_path: str) -> DiseasePrediction:
        """
        Dummy prediction function.
        Replace the body with your real image model later.
        """
        # ------ DUMMY LOGIC (to be replaced) ------
        # Ici on renvoie juste une maladie fixe pour tester le pipeline.
        # Tu peux changer "cystic fibrosis" par une autre pour les tests.
        print(f"[ImageClassifier] Received image: {image_path}")
        print("[ImageClassifier] WARNING: using dummy classifier, not a real model.")
        return DiseasePrediction(
            disease_name="cystic fibrosis",
            confidence=0.5,
        )
        # ------ END DUMMY LOGIC ------


# Helper simple à utiliser partout
def classify_disease_from_image(image_path: str) -> DiseasePrediction:
    classifier = ImageDiseaseClassifier()
    return classifier.predict(image_path)
