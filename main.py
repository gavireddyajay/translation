"""
Translation Models for Commercial Use
1. Helsinki-NLP Opus-MT  — Fast, accurate, per-language-pair
2. Google MADLAD-400     — Single model, 400+ languages
"""

# ============================================================
# Installation (run once)
# ============================================================
# pip install transformers sentencepiece torch accelerate

# ============================================================
# Option 1: Helsinki-NLP Opus-MT (Best for speed + accuracy)
# ============================================================
from transformers import MarianMTModel, MarianTokenizer

class OpusMTTranslator:
    """
    Translates a specific language → English using Opus-MT.
    
    Supported source languages (examples):
      fr, de, es, pt, it, nl, ru, zh, ja, ko, ar, hi, tr,
      pl, sv, fi, da, cs, ro, bg, uk, vi, th, id, ms ...
    
    Full list: https://huggingface.co/Helsinki-NLP
    Naming: Helsinki-NLP/opus-mt-{src}-en
    """

    def __init__(self, src_lang: str = "fr"):
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-en"
        print(f"Loading Opus-MT model: {model_name}")
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.model.eval()

    def translate(self, texts: str | list[str]) -> list[str]:
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        translated = self.model.generate(**inputs)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]


# --- Usage ---
opus = OpusMTTranslator(src_lang="fr")  # French → English
results = opus.translate([
    "Bonjour, comment allez-vous?",
    "La vie est belle quand on est ensemble.",
])
for r in results:
    print(r)


# ============================================================
# Option 2: Google MADLAD-400 (Best for multi-language coverage)
# ============================================================
from transformers import T5ForConditionalGeneration, T5Tokenizer

class MADLAD400Translator:
    """
    Translates 400+ languages → English using a single model.
    
    How it works:
      Prepend <2en> to input text to translate TO English.
      Language is auto-detected from the input text.
    
    Model sizes:
      - google/madlad400-3b-mt   (3B params, ~12GB, best quality)
      - google/madlad400-7b-mt   (7B params, ~28GB)
      - google/madlad400-10b-mt  (10B params, ~40GB)
    """

    def __init__(self, model_size: str = "3b"):
        model_name = f"google/madlad400-{model_size}-mt"
        print(f"Loading MADLAD-400 model: {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()

    def translate(self, texts: str | list[str], target_lang: str = "en") -> list[str]:
        if isinstance(texts, str):
            texts = [texts]
        # Prepend target language tag
        prefixed = [f"<2{target_lang}> {t}" for t in texts]
        inputs = self.tokenizer(prefixed, return_tensors="pt", padding=True, truncation=True)
        translated = self.model.generate(**inputs, max_new_tokens=256)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]


# --- Usage ---
madlad = MADLAD400Translator(model_size="3b")
results = madlad.translate([
    "Bonjour, comment allez-vous?",          # French
    "जीवन एक चॉकलेट बॉक्स की तरह है।",       # Hindi
    "生活就像一盒巧克力。",                       # Chinese
    "Das Leben ist wie eine Schachtel Pralinen.", # German
])
for r in results:
    print(r)


# ============================================================
# Option 3: Hybrid Approach (Recommended for production)
# Use Opus-MT for supported pairs, MADLAD-400 as fallback
# ============================================================

class HybridTranslator:
    """
    Routes to the best model per language:
    - Opus-MT for high-resource pairs (faster, more accurate)
    - MADLAD-400 as fallback for rare languages
    """

    # Languages where Opus-MT models are available and reliable
    OPUS_SUPPORTED = {
        "fr", "de", "es", "pt", "it", "nl", "ru", "zh", "ja",
        "ko", "ar", "hi", "tr", "pl", "sv", "fi", "da", "cs",
        "ro", "bg", "uk", "vi", "th", "id", "ms", "he", "el",
        "hu", "ca", "et", "lt", "lv", "sl", "sk", "hr",
    }

    def __init__(self, madlad_size: str = "3b"):
        self._opus_cache: dict[str, OpusMTTranslator] = {}
        self._madlad = None
        self._madlad_size = madlad_size

    def _get_opus(self, src_lang: str) -> OpusMTTranslator:
        if src_lang not in self._opus_cache:
            self._opus_cache[src_lang] = OpusMTTranslator(src_lang)
        return self._opus_cache[src_lang]

    def _get_madlad(self) -> MADLAD400Translator:
        if self._madlad is None:
            self._madlad = MADLAD400Translator(self._madlad_size)
        return self._madlad

    def translate(self, texts: str | list[str], src_lang: str) -> list[str]:
        if src_lang in self.OPUS_SUPPORTED:
            print(f"Using Opus-MT for '{src_lang}' → en")
            return self._get_opus(src_lang).translate(texts)
        else:
            print(f"Using MADLAD-400 for '{src_lang}' → en")
            return self._get_madlad().translate(texts)


# --- Usage ---
hybrid = HybridTranslator()

# Uses Opus-MT (fast)
print(hybrid.translate("Bonjour le monde!", src_lang="fr"))

# Uses MADLAD-400 (fallback for rare languages)
print(hybrid.translate("Saluton mondo!", src_lang="eo"))  # Esperanto
