"""
utils/speech_converter.py
==========================
Text-to-speech conversion supporting multiple languages.
Uses gTTS (Google Text-to-Speech) for multilingual support
with pyttsx3 as an offline fallback.
"""

import io
import threading
import time

GTTS_AVAILABLE = False
PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
except ImportError:
    pass

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    pass


class SpeechConverter:
    """
    Converts text to speech in a non-blocking background thread.

    Prefers gTTS + pygame for multilingual support. Falls back to
    pyttsx3 for offline/English-only environments.

    Args:
        lang:    BCP-47 language code (e.g. 'en', 'hi', 'ta', 'fr').
        enabled: If False, all speech is silently suppressed.
    """

    def __init__(self, lang: str = "en", enabled: bool = True):
        self.lang = lang
        self.enabled = enabled
        self._speaking = False

        if GTTS_AVAILABLE:
            pygame.mixer.init()
            self._engine = "gtts"
            print(f"[INFO] TTS engine: gTTS (lang={lang})")
        elif PYTTSX3_AVAILABLE:
            self._pyttsx3_engine = pyttsx3.init()
            self._pyttsx3_engine.setProperty("rate", 160)
            self._pyttsx3_engine.setProperty("volume", 1.0)
            self._engine = "pyttsx3"
            print("[INFO] TTS engine: pyttsx3 (offline, English only)")
        else:
            self._engine = None
            print("[WARN] No TTS engine available. Install gtts+pygame or pyttsx3.")

    def speak(self, text: str) -> None:
        """
        Speak the given text in a background thread (non-blocking).

        Args:
            text: The text string to synthesize and play.
        """
        if not self.enabled or not text or self._speaking:
            return
        thread = threading.Thread(target=self._speak_worker, args=(text,), daemon=True)
        thread.start()

    def _speak_worker(self, text: str) -> None:
        """Internal worker that performs TTS synthesis and playback."""
        self._speaking = True
        try:
            if self._engine == "gtts":
                self._speak_gtts(text)
            elif self._engine == "pyttsx3":
                self._speak_pyttsx3(text)
        except Exception as e:
            print(f"[WARN] TTS error: {e}")
        finally:
            self._speaking = False

    def _speak_gtts(self, text: str) -> None:
        tts = gTTS(text=text, lang=self.lang, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        pygame.mixer.music.load(audio_buffer)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)

    def _speak_pyttsx3(self, text: str) -> None:
        self._pyttsx3_engine.say(text)
        self._pyttsx3_engine.runAndWait()
