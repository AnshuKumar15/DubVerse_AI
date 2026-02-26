"""
Translation to Hindi using Facebook NLLB-200 (No Language Left Behind).
Supports 200+ languages → Hindi without requiring HuggingFace authentication.
Lazy-loaded model to avoid blocking imports.
"""

import torch
import json
import os

# Lazy-loaded globals
_tokenizer = None
_model = None
_device = None

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# NLLB language codes: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
NLLB_LANG_CODES = {
    "en": "eng_Latn",    # English
    "hi": "hin_Deva",    # Hindi
    "kn": "kan_Knda",    # Kannada
    "bn": "ben_Beng",    # Bengali
    "ta": "tam_Taml",    # Tamil
    "te": "tel_Telu",    # Telugu
    "ml": "mal_Mlym",    # Malayalam
    "mr": "mar_Deva",    # Marathi
    "gu": "guj_Gujr",    # Gujarati
    "pa": "pan_Guru",    # Punjabi
    "ur": "urd_Arab",    # Urdu
}

# Default: translate from English to Hindi
_source_lang = "en"


def set_source_language(lang_code):
    """Set the source language for translation (ISO 639-1 code, e.g. 'en', 'kn')."""
    global _source_lang
    _source_lang = lang_code
    print(f"  Translation source language set to: {lang_code}")


def _get_device():
    """Pick GPU if enough VRAM is free, otherwise CPU."""
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)  # GB
        # NLLB-600M needs ~2.5 GB; leave some headroom
        if free_mem > 3.0:
            return "cuda"
        else:
            print(f"  GPU has only {free_mem:.1f} GB free — using CPU for translation")
    return "cpu"


def unload_model():
    """Free translation model from memory."""
    global _tokenizer, _model, _device
    if _model is not None:
        del _model
        del _tokenizer
        _model = None
        _tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("  Translation model unloaded (GPU memory freed)")


def _load_model():
    """Lazy-load the NLLB translation model on first use."""
    global _tokenizer, _model, _device
    if _model is not None:
        return

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"  Loading translation model: {MODEL_NAME}")
    _device = _get_device()
    print(f"  Device: {_device}")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    _model.to(_device)
    print("  Translation model loaded.")


def translate_text(text, source_lang=None, target_lang="hi", max_length=512):
    """
    Translate text to Hindi (or any target language).

    Args:
        text: Text to translate.
        source_lang: Source language ISO code (e.g. 'en', 'kn'). Uses global default if None.
        target_lang: Target language ISO code (default: 'hi' for Hindi).
        max_length: Maximum token length for generation.

    Returns:
        Translated text string.
    """
    _load_model()

    src = source_lang or _source_lang
    src_nllb = NLLB_LANG_CODES.get(src, "eng_Latn")
    tgt_nllb = NLLB_LANG_CODES.get(target_lang, "hin_Deva")

    # Set source language for tokenizer
    _tokenizer.src_lang = src_nllb

    batch = _tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(_device)

    # Force target language token
    forced_bos_token_id = _tokenizer.convert_tokens_to_ids(tgt_nllb)

    generated = _model.generate(
        **batch,
        forced_bos_token_id=forced_bos_token_id,
        max_length=max_length,
    )
    output = _tokenizer.batch_decode(generated, skip_special_tokens=True)
    return output[0]


def translate_segments(segments, source_lang=None):
    """
    Translate a list of transcript segments to Hindi.

    Args:
        segments: List of dicts with 'start', 'end', 'text' keys.
        source_lang: Source language code (auto-detected if None).

    Returns:
        List of dicts with 'start', 'end', 'english', 'hindi' keys.
    """
    _load_model()

    translated = []
    total = len(segments)

    for i, seg in enumerate(segments):
        original_text = seg["text"].strip()
        if not original_text:
            continue

        print(f"  Translating segment {i + 1}/{total}: \"{original_text[:60]}...\"")
        hindi_text = translate_text(original_text, source_lang=source_lang)
        print(f"    → {hindi_text[:60]}...")

        translated.append({
            "start": seg["start"],
            "end": seg["end"],
            "english": original_text,
            "hindi": hindi_text,
        })

    return translated


def save_translated_transcript(translated_segments, output_path):
    """Save translated segments to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"segments": translated_segments}, f, indent=4, ensure_ascii=False)
    print(f"  Hindi transcript saved → {output_path}")


def load_translated_transcript(input_path):
    """Load a previously saved Hindi transcript."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["segments"]