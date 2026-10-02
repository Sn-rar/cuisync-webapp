import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

# 1. Suppress low-level Paddle & PaddleX logs BEFORE initializing engines
os.environ["GLOG_minloglevel"] = "3"           # Suppress Paddle C++ logs (0=INFO, 1=WARN, 2=ERR, 3=FATAL)
os.environ["PADDLE_PDX_LOG_LEVEL"] = "ERROR"   # Suppress PaddleX pipeline inspection logs

logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddlex").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)

import cv2
import numpy as np
from flask import has_app_context
from paddleocr import PaddleOCR

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
VENDOR_DIR = os.path.join(ROOT_DIR, ".vendor")
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

try:
    from unidecode import unidecode
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "The 'unidecode' package is required for OCR romanization but is not installed in the active environment."
    ) from None

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "The 'indic-transliteration' package is required for Devanagari romanization but is not installed in the active environment."
    ) from None

# Japanese Hepburn Romanizer
try:
    import pykakasi
    _kakasi = pykakasi.kakasi()
except ModuleNotFoundError:
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".vendor"))
        import pykakasi
        _kakasi = pykakasi.kakasi()
    except ModuleNotFoundError:
        _kakasi = None

logger = logging.getLogger(__name__)

COUNTRY_LANG_MAP = {
    "cambodia": "km",
    "china": "ch",
    "india": "hi",
    "indonesia": "id",
    "japan": "japan",
    "malaysia": "ms",
    "myanmar": "my",
    "philippines": "tl",
    "saudi arabia": "ar",
    "south korea": "korean",
    "thailand": "th",
    "turkiye": "tr",
    "turkey": "tr",
    "vietnam": "vi",
}

OCR_ENGINES = {}
_ENGINE_LOCK = threading.Lock()

UNSUPPORTED_LANGS = frozenset({"km", "my"})

# In PP-OCRv5, 'tl', 'id', 'ms' all route to 'latin_PP-OCRv5_mobile_rec'.
# Running distinct distinct script families avoids repeating the exact same model.
AUTO_OCR_LANGUAGES = (
    "japan",   # Japanese CJK / Kana
    "ch",      # Simplified Chinese Hanzi
    "korean",  # Hangul
    "th",      # Thai
    "hi",      # Devanagari / Hindi
    "ar",      # Arabic
    "vi",      # Vietnamese (special Latin tonal marks)
    "en",      # General Latin (covers English, Tagalog, Indonesian, Malay)
)

OCR_WORKER_COUNT = max(2, min(len(AUTO_OCR_LANGUAGES), os.cpu_count() or 4))

NON_ASCII_BONUS_MULTIPLIER = 1.5

# Script-specific Unicode character blocks
KANA_RANGE = re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")       # Japanese Hiragana & Katakana
CJK_RANGE = re.compile(r"[\u4E00-\u9FFF]")                      # CJK Unified Ideographs (Kanji / Hanzi)
HANGUL_RANGE = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF]")     # Korean Hangul
THAI_RANGE = re.compile(r"[\u0E00-\u0E7F]")                     # Thai
KHMER_RANGE = re.compile(r"[\u1780-\u17FF]")                    # Cambodian / Khmer
MYANMAR_RANGE = re.compile(r"[\u1000-\u109F]")                  # Burmese / Myanmar
SEA_RANGE = re.compile(r"[\u0E00-\u0E7F\u1780-\u17FF\u1000-\u109F]")  # Thailand, Cambodia, Myanmar combined
ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")                   # Arabic
DEVANAGARI_RANGE = re.compile(r"[\u0900-\u097F]")               # Hindi / Devanagari

MIN_SUBSTRING_MATCH_LEN = 4
MAX_CANDIDATE_LINES = 60


def get_logger():
    if has_app_context():
        from flask import current_app
        return current_app.logger
    return logger


def get_ocr_for_language(lang):
    """Load and cache a PaddleOCR engine for one supported language (thread-safe)."""
    if lang in UNSUPPORTED_LANGS:
        raise ValueError(
            f"PaddleOCR has no model for lang={lang!r} in any version. "
            "Khmer and Burmese require a different OCR backend."
        )
    engine = OCR_ENGINES.get(lang)
    if engine is None:
        with _ENGINE_LOCK:
            engine = OCR_ENGINES.get(lang)
            if engine is None:
                engine = PaddleOCR(
                    lang=lang,
                    ocr_version="PP-OCRv5",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    text_det_limit_side_len=960,
                    text_det_limit_type="max",
                    text_recognition_batch_size=16,
                    show_log=False,
                )
                OCR_ENGINES[lang] = engine
    return engine


def warmup_ocr_engines():
    """Call this once when your app initializes to preload all models ahead of requests."""
    for lang in AUTO_OCR_LANGUAGES:
        get_ocr_for_language(lang)


def get_ocr(country=""):
    """Load the OCR engine mapped to a country, defaulting to English."""
    lang = COUNTRY_LANG_MAP.get((country or "").strip().lower(), "en")
    return get_ocr_for_language(lang)


def result_data(result):
    """Return the serializable data from a PaddleOCR 3.x OCRResult."""
    data = result.json
    data = data() if callable(data) else data
    if isinstance(data, str):
        return json.loads(data)
    if isinstance(data, dict):
        return data
    raise RuntimeError("PaddleOCR returned an unreadable result.")


def extract_lines_with_scores(image, ocr_model):
    """Return recognized lines and PaddleOCR confidence scores."""
    lines = []
    scores = []
    for result in ocr_model.predict(image):
        data = result_data(result)
        payload = data.get("res", data)
        result_scores = payload.get("rec_scores", [])
        for index, text in enumerate(payload.get("rec_texts", [])):
            text = str(text).strip()
            if text:
                lines.append(text)
                try:
                    scores.append(float(result_scores[index]))
                except (IndexError, TypeError, ValueError):
                    scores.append(0.5)
    return lines, scores


def ocr_quality(lines, scores, language=""):
    """Prefer confident results that recognize more meaningful text."""
    base = sum(max(score, 0.0) * len(line) for line, score in zip(lines, scores))
    full_text = "".join(lines)
    has_kana = bool(KANA_RANGE.search(full_text))
    has_cjk = bool(CJK_RANGE.search(full_text))

    if language == "japan" and has_kana:
        base *= 3.0
    elif language == "ch" and has_cjk and not has_kana:
        base *= 1.8
    elif language == "th" and THAI_RANGE.search(full_text):
        base *= 3.0
    elif language == "korean" and HANGUL_RANGE.search(full_text):
        base *= 3.0
    elif language == "hi" and DEVANAGARI_RANGE.search(full_text):
        base *= 3.0
    elif language == "ar" and ARABIC_RANGE.search(full_text):
        base *= 3.0
    elif language != "en" and any(ord(ch) > 127 for ch in full_text):
        base *= NON_ASCII_BONUS_MULTIPLIER

    return base


def _run_language_ocr(language, image):
    """Run one language's OCR engine without crashing concurrent execution."""
    try:
        lines, scores = extract_lines_with_scores(
            image, get_ocr_for_language(language)
        )
    except Exception:
        get_logger().exception("OCR model '%s' failed, skipping", language)
        return language, [], []
    return language, lines, scores


def automatic_ocr(image):
    """Detect text and merge related script families."""
    results = {}
    with ThreadPoolExecutor(max_workers=OCR_WORKER_COUNT) as executor:
        futures = [
            executor.submit(_run_language_ocr, language, image)
            for language in AUTO_OCR_LANGUAGES
        ]
        for future in as_completed(futures):
            language, lines, scores = future.result()
            results[language] = (lines, scores)

    best_lines = []
    best_score = -1.0
    best_language = "en"
    cjk_pool = {}
    sea_pool = {}

    for language in AUTO_OCR_LANGUAGES:
        lines, scores = results.get(language, ([], []))
        full_text = "".join(lines)

        if language in ("japan", "ch") and CJK_RANGE.search(full_text):
            for line in lines:
                cjk_pool[line] = max(cjk_pool.get(line, 0), len(line))

        if language == "th" and SEA_RANGE.search(full_text):
            for line in lines:
                sea_pool[line] = max(sea_pool.get(line, 0), len(line))

        score = ocr_quality(lines, scores, language)
        if score > best_score:
            best_lines = lines
            best_score = score
            best_language = language

    if cjk_pool and (best_language in ("japan", "ch")):
        merged = list(dict.fromkeys(best_lines + list(cjk_pool.keys())))
        get_logger().info("OCR extracted text (CJK combined): %s", merged)
        return merged

    if sea_pool and best_language == "th":
        merged = list(dict.fromkeys(best_lines + list(sea_pool.keys())))
        get_logger().info("OCR extracted text (Southeast Asia combined): %s", merged)
        return merged

    get_logger().info("OCR extracted text (%s): %s", best_language, best_lines)
    return best_lines


def romanize_japanese(text):
    """Convert Japanese Kanji and Kana to Hepburn Romaji using pykakasi."""
    if not _kakasi:
        return text
    result = _kakasi.convert(str(text))
    return " ".join(item["hepburn"] for item in result if item.get("hepburn")).strip()


def romanize_text(text, mode="standard"):
    """Convert non-Latin writing to Latin characters without translating."""
    text = str(text)
    if DEVANAGARI_RANGE.search(text):
        text = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    elif mode == "japanese" and _kakasi:
        text = romanize_japanese(text)
    else:
        text = unidecode(text)
    return re.sub(r"\s+", " ", text).strip()


def romanize_lines(lines):
    """Romanize OCR text and log readings for CJK and Southeast Asian scripts."""
    romanized_std = [romanize_text(line, mode="standard") for line in lines if str(line).strip()]

    has_cjk = any(CJK_RANGE.search(str(line)) or KANA_RANGE.search(str(line)) for line in lines)
    if has_cjk:
        romanized_ja = [romanize_text(line, mode="japanese") for line in lines if str(line).strip()]
        get_logger().info("Romanized text (Standard/Pinyin): %s", " ".join(romanized_std))
        get_logger().info("Romanized text (Japanese Hepburn): %s", " ".join(romanized_ja))
        return list(dict.fromkeys(romanized_std + romanized_ja))

    has_sea = any(SEA_RANGE.search(str(line)) for line in lines)
    if has_sea:
        romanized_sea = [romanize_text(line, mode="standard") for line in lines if str(line).strip()]
        get_logger().info("Romanized text (Southeast Asia - TH/KM/MY): %s", " ".join(romanized_sea))
        return list(dict.fromkeys(romanized_std + romanized_sea))

    get_logger().info("Romanized text: %s", " ".join(romanized_std))
    return romanized_std


def match_key(text, mode="standard"):
    """Create a punctuation- and accent-insensitive key for recipe matching."""
    return re.sub(r"[^a-z0-9]+", "", romanize_text(text, mode=mode).casefold())


def menu_candidates(romanized_lines):
    """Build complete dish candidates from romanized OCR fragments."""
    raw_keys = []

    for line in romanized_lines:
        line_str = str(line).strip()
        if not line_str:
            continue

        key_std = match_key(line_str, mode="standard")
        if key_std:
            raw_keys.append(key_std)

        if CJK_RANGE.search(line_str) or KANA_RANGE.search(line_str):
            key_ja = match_key(line_str, mode="japanese")
            if key_ja and key_ja != key_std:
                raw_keys.append(key_ja)

        if SEA_RANGE.search(line_str):
            key_sea = match_key(line_str, mode="standard")
            if key_sea and key_sea != key_std:
                raw_keys.append(key_sea)

    raw_keys = raw_keys[:MAX_CANDIDATE_LINES]
    candidates = set(raw_keys)

    for start in range(len(raw_keys)):
        joined = ""
        for end in range(start, min(start + 10, len(raw_keys))):
            joined += raw_keys[end]
            candidates.add(joined)

    if raw_keys and len(raw_keys) <= 12:
        candidates.add("".join(raw_keys))
    return candidates


def build_candidate_index(candidate_keys):
    """Precompute a fast search string plus length buckets, once per request."""
    search_string = "|" + "|".join(candidate_keys) + "|"
    length_buckets = {}
    for candidate in candidate_keys:
        length_buckets.setdefault(len(candidate), []).append(candidate)
    return search_string, length_buckets


def is_recipe_match(title_key, candidate_search_string, length_buckets):
    """Match exact menu text, tolerating a small OCR typo in a full title."""
    if len(title_key) >= MIN_SUBSTRING_MATCH_LEN and title_key in candidate_search_string:
        return True

    if len(title_key) >= 8:
        for length in range(len(title_key) - 4, len(title_key) + 5):
            for candidate in length_buckets.get(length, ()):
                if SequenceMatcher(None, title_key, candidate).ratio() >= 0.84:
                    return True
    return False


def title_key_variants(title, alternative_title="", country=""):
    """Return every matchable key for a recipe title without splitting on dashes."""
    is_japan = (country or "").strip().lower() in ("japan", "japanese")
    raw_names = [title, alternative_title]
    keys = []
    seen = set()

    for name in raw_names:
        if not name:
            continue

        if is_japan:
            k_ja = match_key(name, mode="japanese")
            if k_ja and k_ja not in seen:
                keys.append(k_ja)
                seen.add(k_ja)

        k_std = match_key(name, mode="standard")
        if k_std and k_std not in seen:
            keys.append(k_std)
            seen.add(k_std)

    return keys


def find_recipes(romanized_lines, origin_country="", recipes=None):
    """Return every database recipe recognized in a menu, without translating."""
    if recipes is None:
        recipes = []

    candidate_keys = menu_candidates(romanized_lines)
    candidate_search_string, length_buckets = build_candidate_index(candidate_keys)
    origin_key = origin_country.strip().casefold()
    matches = []
    seen = set()

    recipe_title_key_variants = [
        title_key_variants(
            recipe.get("title", ""),
            recipe.get("alternative_title", ""),
            country=recipe.get("country", "")
        )
        for recipe in recipes
    ]
    recipe_country_keys = [
        recipe.get("country", "").strip().casefold() for recipe in recipes
    ]

    for index, recipe in enumerate(recipes):
        title = recipe.get("title", "").strip()
        if not title or (origin_key and recipe_country_keys[index] != origin_key):
            continue

        title_keys = recipe_title_key_variants[index]
        if not title_keys:
            continue

        if any(
            is_recipe_match(title_key, candidate_search_string, length_buckets)
            for title_key in title_keys
        ):
            country = recipe.get("country", "").strip()
            identity = (title.casefold(), country.casefold())
            if identity not in seen:
                matches.append({
                    "title": title,
                    "country": country,
                    "alternative_title": recipe.get("alternative_title", "").strip(),
                    "image_link": recipe.get("image_link", "").strip(),
                })
                seen.add(identity)

    return [
        match
        for match in matches
        if not any(
            match is not other
            and match["country"].casefold() == other["country"].casefold()
            and match_key(match["title"]) != match_key(other["title"])
            and match_key(match["title"]) in match_key(other["title"])
            for other in matches
        )
    ]