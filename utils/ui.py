"""
utils/ui.py
===========
User interface overlay rendering for the Sign Language Bridge system.
Draws gesture labels, confidence bars, sentence buffers, and the
text-to-sign panel onto the live video frame.
"""

import cv2
import numpy as np


# UI color palette
COLOR_GREEN  = (0, 220, 80)
COLOR_YELLOW = (0, 210, 255)
COLOR_RED    = (0, 80, 220)
COLOR_WHITE  = (240, 240, 240)
COLOR_DARK   = (20, 20, 20)
COLOR_PANEL  = (35, 35, 35)


class SignLanguageUI:
    """
    Renders all UI overlays onto the video frame.

    Args:
        lang: Active output language code for display.
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self._text_input: str = ""

    def get_text_input(self) -> str:
        """Return the current text-to-sign input string."""
        return self._text_input

    def set_text_input(self, text: str) -> None:
        """Update the text-to-sign input."""
        self._text_input = text.strip()

    def render(
        self,
        frame: np.ndarray,
        gesture: str | None,
        confidence: float,
        buffer: list[str],
        sign_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compose all UI elements onto the frame.

        Args:
            frame:      Annotated BGR video frame.
            gesture:    Currently detected gesture label (or None).
            confidence: Classifier confidence score (0.0–1.0).
            buffer:     List of recognized words accumulated this session.
            sign_frame: Optional text-to-sign panel image.

        Returns:
            Final composited BGR frame ready for display.
        """
        h, w = frame.shape[:2]

        # Top status bar
        self._draw_status_bar(frame, w, gesture, confidence)

        # Sentence buffer strip
        self._draw_sentence_buffer(frame, h, w, buffer)

        # Text-to-sign panel (bottom right)
        if sign_frame is not None:
            self._overlay_sign_panel(frame, h, w, sign_frame)

        # Keyboard hint strip
        self._draw_hints(frame, h, w)

        return frame

    # ------------------------------------------------------------------
    # Private rendering helpers
    # ------------------------------------------------------------------

    def _draw_status_bar(
        self,
        frame: np.ndarray,
        w: int,
        gesture: str | None,
        confidence: float,
    ) -> None:
        """Draw the top status bar with gesture label and confidence."""
        bar_h = 52
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), COLOR_DARK, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Title
        cv2.putText(frame, "Sign Language Bridge", (12, 34),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, COLOR_GREEN, 2)

        # Language tag
        lang_text = f"[{self.lang.upper()}]"
        cv2.putText(frame, lang_text, (w - 80, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)

        # Gesture label
        if gesture:
            label = f"Detected: {gesture}"
            cv2.putText(frame, label, (260, 34),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, COLOR_WHITE, 2)

            # Confidence bar
            bar_x, bar_y = 500, 20
            bar_w_max = 150
            filled = int(bar_w_max * confidence)
            bar_color = COLOR_GREEN if confidence >= 0.8 else COLOR_YELLOW
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_max, bar_y + 14),
                          (80, 80, 80), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 14),
                          bar_color, -1)
            cv2.putText(frame, f"{confidence:.0%}", (bar_x + bar_w_max + 6, bar_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

    def _draw_sentence_buffer(
        self,
        frame: np.ndarray,
        h: int,
        w: int,
        buffer: list[str],
    ) -> None:
        """Draw the accumulated recognized sentence at the bottom left."""
        if not buffer:
            return

        sentence = " ".join(buffer[-8:])  # Show last 8 words
        strip_y = h - 55
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, strip_y), (w // 2 + 100, h - 30), COLOR_PANEL, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "Buffer:", (10, strip_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1)
        cv2.putText(frame, sentence, (72, strip_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

    def _overlay_sign_panel(
        self,
        frame: np.ndarray,
        h: int,
        w: int,
        sign_frame: np.ndarray,
    ) -> None:
        """Embed the text-to-sign panel in the bottom-right corner."""
        ph, pw = sign_frame.shape[:2]
        x_off = w - pw - 10
        y_off = h - ph - 40

        # Border
        cv2.rectangle(frame, (x_off - 2, y_off - 2),
                      (x_off + pw + 2, y_off + ph + 2), COLOR_GREEN, 2)

        frame[y_off:y_off + ph, x_off:x_off + pw] = sign_frame

        cv2.putText(frame, "Text → Sign", (x_off, y_off - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GREEN, 1)

    def _draw_hints(self, frame: np.ndarray, h: int, w: int) -> None:
        """Draw keyboard shortcut hints at the very bottom."""
        hints = "  [Q] Quit    [S] Speak Buffer    [C] Clear Buffer"
        cv2.putText(frame, hints, (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)
