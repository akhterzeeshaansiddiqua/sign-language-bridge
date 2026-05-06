"""
utils/hand_detection.py
=======================
Real-time hand landmark detection using MediaPipe Hands.
Extracts 21 3D landmarks per hand for gesture classification.
"""

import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    """
    Detects and tracks hand landmarks in video frames using MediaPipe Hands.

    Attributes:
        max_hands:   Maximum number of hands to detect simultaneously.
        min_det_conf: Minimum confidence for initial hand detection.
        min_track_conf: Minimum confidence to continue tracking a hand.
    """

    # MediaPipe hand connection pairs for drawing skeleton
    LANDMARK_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

    def __init__(
        self,
        max_hands: int = 1,
        min_det_conf: float = 0.7,
        min_track_conf: float = 0.6,
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )

    def detect(self, frame: np.ndarray) -> tuple[list | None, np.ndarray]:
        """
        Run hand detection on a single BGR frame.

        Args:
            frame: Input BGR video frame from OpenCV.

        Returns:
            Tuple of:
              - landmarks: Flattened list of 63 floats (21 landmarks × [x, y, z]),
                           normalized to wrist position; None if no hand detected.
              - annotated:  Frame with hand skeleton drawn.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        annotated = frame.copy()
        landmarks = None

        if results.multi_hand_landmarks:
            hand_lm = results.multi_hand_landmarks[0]  # Use first detected hand

            # Draw landmarks and connections
            self.mp_draw.draw_landmarks(
                annotated,
                hand_lm,
                self.LANDMARK_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )

            landmarks = self._extract_landmarks(hand_lm)

        return landmarks, annotated

    def _extract_landmarks(self, hand_landmarks) -> list[float]:
        """
        Extract and normalize landmark coordinates relative to the wrist (landmark 0).

        Normalization makes the feature vector scale- and position-invariant,
        improving classifier generalization.

        Args:
            hand_landmarks: MediaPipe NormalizedLandmarkList object.

        Returns:
            Flattened list of 63 normalized floats [x0,y0,z0, x1,y1,z1, ...].
        """
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
        )

        # Normalize relative to wrist (landmark index 0)
        wrist = coords[0]
        coords -= wrist

        # Scale by the maximum absolute value to make unit-independent
        scale = np.max(np.abs(coords)) + 1e-6
        coords /= scale

        return coords.flatten().tolist()
