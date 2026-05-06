"""
utils/sign_display.py
======================
Text-to-Sign Language visual output module.
Looks up pre-recorded sign video clips or static gesture images
for each word in the input text and returns the current display frame.
"""

import os
import cv2
import numpy as np


# Default directory structure for sign assets
SIGN_VIDEO_DIR = "assets/signs/videos"
SIGN_IMAGE_DIR = "assets/signs/images"

# Target display size for sign panels
SIGN_PANEL_SIZE = (320, 240)


class SignDisplay:
    """
    Provides visual sign language output for text input.

    Looks up each word in a library of pre-recorded video clips (.mp4)
    or static images (.png / .jpg). Renders the current frame for display.

    Falls back to a text placeholder if no asset is found for a word.
    """

    def __init__(
        self,
        video_dir: str = SIGN_VIDEO_DIR,
        image_dir: str = SIGN_IMAGE_DIR,
    ):
        self.video_dir = video_dir
        self.image_dir = image_dir
        self._current_cap = None
        self._current_word = None
        self._word_queue: list[str] = []
        self._queue_index = 0

    def get_sign_frame(self, text: str) -> np.ndarray:
        """
        Return the next video frame or static image for the given text.

        Words are displayed sequentially — one word per call cycle.

        Args:
            text: Input text string (e.g. "hello please stop").

        Returns:
            BGR image array (SIGN_PANEL_SIZE) for display, or a placeholder.
        """
        words = text.strip().lower().split()
        if not words:
            return self._placeholder("(no input)")

        # Reset queue when text changes
        if words != self._word_queue:
            self._word_queue = words
            self._queue_index = 0
            self._release_cap()

        word = self._word_queue[self._queue_index % len(self._word_queue)]

        frame = self._try_video(word)
        if frame is None:
            frame = self._try_image(word)
        if frame is None:
            frame = self._placeholder(word.upper())

        return frame

    def advance_word(self) -> None:
        """Move to the next word in the queue."""
        self._queue_index += 1
        self._release_cap()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_video(self, word: str) -> np.ndarray | None:
        """Attempt to read the next frame from a sign video clip."""
        if self._current_word != word:
            self._release_cap()
            path = os.path.join(self.video_dir, f"{word}.mp4")
            if os.path.exists(path):
                self._current_cap = cv2.VideoCapture(path)
                self._current_word = word

        if self._current_cap and self._current_cap.isOpened():
            ret, frame = self._current_cap.read()
            if ret:
                return cv2.resize(frame, SIGN_PANEL_SIZE)
            else:
                # Video ended — loop or advance
                self._current_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return None

        return None

    def _try_image(self, word: str) -> np.ndarray | None:
        """Attempt to load a static sign image for the word."""
        for ext in (".png", ".jpg", ".jpeg"):
            path = os.path.join(self.image_dir, f"{word}{ext}")
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    return cv2.resize(img, SIGN_PANEL_SIZE)
        return None

    def _placeholder(self, word: str) -> np.ndarray:
        """Generate a dark placeholder panel with the word as text."""
        panel = np.zeros((SIGN_PANEL_SIZE[1], SIGN_PANEL_SIZE[0], 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        cv2.putText(
            panel,
            word,
            (20, SIGN_PANEL_SIZE[1] // 2),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 100),
            2,
        )
        cv2.putText(
            panel,
            "Sign video not found",
            (10, SIGN_PANEL_SIZE[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (120, 120, 120),
            1,
        )
        return panel

    def _release_cap(self) -> None:
        if self._current_cap:
            self._current_cap.release()
            self._current_cap = None
            self._current_word = None
