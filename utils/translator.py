"""
utils/translator.py
====================
Multilingual text translation using Google Translate API (via googletrans).
Falls back to returning the original text if translation fails or
the library is unavailable.
"""

GOOGLETRANS_AVAILABLE = False
try:
    from googletrans import Translator as GoogleTranslator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    pass

# Supported language codes and display names
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ar": "Arabic",
    "zh-cn": "Chinese (Simplified)",
    "ja": "Japanese",
}


class Translator:
    """
    Translates recognized gesture labels into the target language.

    Args:
        target_lang: BCP-47 language code for translation output.
    """

    def __init__(self, target_lang: str = "en"):
        self.target_lang = target_lang
        self._client = None

        if target_lang == "en":
            return  # No translation needed

        if GOOGLETRANS_AVAILABLE:
            self._client = GoogleTranslator()
            lang_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)
            print(f"[INFO] Translator active: English → {lang_name}")
        else:
            print(
                "[WARN] googletrans not installed. Output will remain in English.\n"
                "       Install with: pip install googletrans==4.0.0-rc1"
            )

    def translate(self, text: str) -> str:
        """
        Translate text from English to the configured target language.

        Args:
            text: English text to translate (e.g. a recognized gesture label).

        Returns:
            Translated string, or original text if translation is unavailable.
        """
        if self._client is None or self.target_lang == "en":
            return text

        try:
            result = self._client.translate(text, src="en", dest=self.target_lang)
            return result.text
        except Exception as e:
            print(f"[WARN] Translation failed for '{text}': {e}")
            return text

    @staticmethod
    def list_languages() -> None:
        """Print all supported language codes."""
        print("\nSupported Languages:")
        for code, name in SUPPORTED_LANGUAGES.items():
            print(f"  {code:10s} → {name}")
