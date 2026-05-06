"""
utils/gesture_recognizer.py
============================
ML-based gesture classifier using landmark features from MediaPipe.
Supports scikit-learn models (RandomForest, SVM, MLP) loaded from disk,
with a fallback rule-based recognizer for common static gestures.
"""

import os
import pickle
import numpy as np


# Rule-based fallback labels (used when no trained model is found)
RULE_BASED_GESTURES = {
    "hello": "Hello",
    "yes": "Yes",
    "no": "No",
    "thank_you": "Thank You",
    "please": "Please",
    "help": "Help",
    "stop": "Stop",
    "good": "Good",
    "bad": "Bad",
    "water": "Water",
    "food": "Food",
    "home": "Home",
}


class GestureRecognizer:
    """
    Classifies hand gestures from landmark feature vectors.

    Uses a pre-trained scikit-learn model if available; otherwise falls back
    to a simple rule-based system for demonstration purposes.

    Args:
        model_path:  Path to a pickled scikit-learn classifier.
        confidence:  Minimum probability threshold to accept a prediction.
    """

    def __init__(self, model_path: str = "models/gesture_model.pkl", confidence: float = 0.75):
        self.confidence = confidence
        self.model = None
        self.labels = list(RULE_BASED_GESTURES.values())

        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    saved = pickle.load(f)
                    self.model = saved.get("model")
                    self.labels = saved.get("labels", self.labels)
                print(f"[INFO] Gesture model loaded from {model_path}")
                print(f"[INFO] Classes: {self.labels}")
            except Exception as e:
                print(f"[WARN] Could not load model: {e} — using rule-based fallback")
        else:
            print("[WARN] No trained model found. Using rule-based fallback.")
            print("[TIP]  Run `python train.py` to train a model on your own data.")

    def predict(self, landmarks: list[float]) -> tuple[str | None, float]:
        """
        Predict the gesture label from a landmark feature vector.

        Args:
            landmarks: 63-float normalized landmark list from HandDetector.

        Returns:
            Tuple of (label, confidence). Returns (None, 0.0) if below threshold.
        """
        if self.model is not None:
            return self._ml_predict(landmarks)
        return self._rule_predict(landmarks)

    def _ml_predict(self, landmarks: list[float]) -> tuple[str | None, float]:
        """Use the trained scikit-learn classifier."""
        X = np.array(landmarks).reshape(1, -1)
        proba = self.model.predict_proba(X)[0]
        best_idx = int(np.argmax(proba))
        best_conf = float(proba[best_idx])

        if best_conf >= self.confidence:
            return self.labels[best_idx], best_conf
        return None, best_conf

    def _rule_predict(self, landmarks: list[float]) -> tuple[str | None, float]:
        """
        Simple geometric rule-based fallback for a few static gestures.
        Returns a dummy confidence of 0.85 when a rule fires.
        """
        coords = np.array(landmarks).reshape(21, 3)

        # Tip indices: thumb=4, index=8, middle=12, ring=16, pinky=20
        # Base indices: thumb=2, index=5, middle=9, ring=13, pinky=17
        tips = [4, 8, 12, 16, 20]
        bases = [2, 5, 9, 13, 17]

        fingers_up = [
            coords[tips[i]][1] < coords[bases[i]][1] for i in range(5)
        ]

        # All fingers up → "Hello"
        if all(fingers_up):
            return "Hello", 0.85

        # All fingers down (fist) → "Stop" / "No"
        if not any(fingers_up):
            return "Stop", 0.85

        # Only index up → pointing / "Yes"
        if fingers_up[1] and not any(fingers_up[i] for i in [0, 2, 3, 4]):
            return "Yes", 0.85

        # Index + middle up → "Good" / peace
        if fingers_up[1] and fingers_up[2] and not fingers_up[3] and not fingers_up[4]:
            return "Good", 0.85

        return None, 0.0
