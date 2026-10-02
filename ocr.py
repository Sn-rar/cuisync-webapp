import json
import logging
import os
import re
import sys
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

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
# Guards lazy engine creation so two threads can't build the same language's
# PaddleOCR instance at once (wasteful, and PaddleOCR init isn't guaranteed
# thread-safe on its own).
_ENGINE_LOCK = threading.Lock()

AUTO_OCR_LANGUAGES = (
    "japan",
    "ch",
    "th",
    "km",
    "my",
    "korean",
    "hi",
    "ar",
    *(
        lang
        for lang in sorted(set(COUNTRY_LANG_MAP.values()))
        if lang not in {"japan", "ch", "th", "km", "my", "korean", "hi", "ar"}
    ),
    "en",
)

# Caps how many language engines run at once. Running all of them
# concurrently (rather than one after another) is the main speed win here:
# total wall-clock time becomes roughly "the slowest single engine" instead
# of "the sum of every engine," since Paddle's inference releases the GIL
# during the actual forward pass. Capped by CPU count so we don't oversubscribe.
OCR_WORKER_COUNT = 1

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
    engine = OCR_ENGINES.get(lang)
    if engine is None:
        with _ENGINE_LOCK:
            engine = OCR_ENGINES.get(lang)
            if engine is None:
                engine = PaddleOCR(
                    lang=lang,
                    ocr_version="PP-OCRv3",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    text_det_limit_side_len=640,
                    text_det_limit_type="max",
                    text_recognition_batch_size=1,
                )
                OCR_ENGINES[lang] = engine
    return engine


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
    elif language == "km" and KHMER_RANGE.search(full_text):
        base *= 3.0
    elif language == "my" and MYANMAR_RANGE.search(full_text):
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
    """Run one language's OCR engine; never raises, so one bad engine can't
    sink the whole concurrent batch."""
    try:
        lines, scores = extract_lines_with_scores(
            image, get_ocr_for_language(language)
        )
    except Exception:
        get_logger().exception("OCR model '%s' failed, skipping", language)
        return language, [], []
    return language, lines, scores


def automatic_ocr(image):
    """Detect text using supported OCR languages concurrently.

    Uses a small worker pool to avoid CPU/RAM oversubscription while retaining
    the original multi-language detection behavior.
    """
    results = {}

    # Avoid executor overhead when the configuration contains only one language.
    if len(AUTO_OCR_LANGUAGES) == 1:
        language = AUTO_OCR_LANGUAGES[0]
        _, lines, scores = _run_language_ocr(language, image)
        results[language] = (lines, scores)
    else:
        with ThreadPoolExecutor(max_workers=OCR_WORKER_COUNT) as executor:
            futures = {
                executor.submit(_run_language_ocr, language, image): language
                for language in AUTO_OCR_LANGUAGES
            }
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

        if language in ("th", "km", "my") and SEA_RANGE.search(full_text):
            for line in lines:
                sea_pool[line] = max(sea_pool.get(line, 0), len(line))

        score = ocr_quality(lines, scores, language)
        if score > best_score:
            best_lines = lines
            best_score = score
            best_language = language

    if cjk_pool and best_language in ("japan", "ch"):
        return list(dict.fromkeys(best_lines + list(cjk_pool.keys())))

    if sea_pool and best_language in ("th", "km", "my"):
        return list(dict.fromkeys(best_lines + list(sea_pool.keys())))

    return best_lines

def romanize_japanese(text):
    """Convert Japanese Kanji and Kana to Hepburn Romaji using pykakasi."""
    if not _kakasi:
        return text
    result = _kakasi.convert(str(text))
    return " ".join(item["hepburn"] for item in result if item.get("hepburn")).strip()


@lru_cache(maxsize=8192)
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

    # 1. CJK / Japanese Check
    has_cjk = any(CJK_RANGE.search(str(line)) or KANA_RANGE.search(str(line)) for line in lines)
    if has_cjk:
        romanized_ja = [romanize_text(line, mode="japanese") for line in lines if str(line).strip()]
        get_logger().info("Romanized text (Standard/Pinyin): %s", " ".join(romanized_std))
        get_logger().info("Romanized text (Japanese Hepburn): %s", " ".join(romanized_ja))
        return list(dict.fromkeys(romanized_std + romanized_ja))

    # 2. Southeast Asian Check: Thailand, Cambodia, Myanmar
    has_sea = any(SEA_RANGE.search(str(line)) for line in lines)
    if has_sea:
        romanized_sea = [romanize_text(line, mode="standard") for line in lines if str(line).strip()]
        get_logger().info("Romanized text (Southeast Asia - TH/KM/MY): %s", " ".join(romanized_sea))
        return list(dict.fromkeys(romanized_std + romanized_sea))

    get_logger().info("Romanized text: %s", " ".join(romanized_std))
    return romanized_std


@lru_cache(maxsize=16384)
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

        # CJK Candidate Key
        if CJK_RANGE.search(line_str) or KANA_RANGE.search(line_str):
            key_ja = match_key(line_str, mode="japanese")
            if key_ja and key_ja != key_std:
                raw_keys.append(key_ja)

        # Southeast Asian (Thailand, Cambodia, Myanmar) Candidate Key
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


@lru_cache(maxsize=16384)
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
    if not recipes:
        return []

    candidate_keys = menu_candidates(romanized_lines)
    if not candidate_keys:
        return []

    candidate_search_string, length_buckets = build_candidate_index(candidate_keys)
    origin_key = origin_country.strip().casefold()

    matches = []
    seen = set()

    # Filter the recipe list first. This avoids doing romanization/key generation
    # for recipes that can never match the requested country.
    filtered_recipes = []
    for recipe in recipes:
        title = recipe.get("title", "").strip()
        if not title:
            continue

        country = recipe.get("country", "").strip()
        if origin_key and country.casefold() != origin_key:
            continue

        filtered_recipes.append((recipe, title, country))

    for recipe, title, country in filtered_recipes:
        title_keys = title_key_variants(
            title,
            recipe.get("alternative_title", ""),
            country=country,
        )
        if not title_keys:
            continue

        if any(
            is_recipe_match(title_key, candidate_search_string, length_buckets)
            for title_key in title_keys
        ):
            identity = (title.casefold(), country.casefold())
            if identity not in seen:
                matches.append({
                    "title": title,
                    "country": country,
                    "alternative_title": recipe.get("alternative_title", "").strip(),
                    "image_link": recipe.get("image_link", "").strip(),
                    "_title_key": match_key(title),
                })
                seen.add(identity)

    # Remove shorter duplicate/substrings without repeatedly recalculating keys.
    result = []
    for match in matches:
        title_key = match["_title_key"]
        country_key = match["country"].casefold()

        is_redundant = any(
            match is not other
            and country_key == other["country"].casefold()
            and title_key != other["_title_key"]
            and title_key in other["_title_key"]
            for other in matches
        )

        if not is_redundant:
            match.pop("_title_key", None)
            result.append(match)

    return result


# Optional fast path: when the country is already known, OCR only the mapped
# language instead of loading/running every supported language.
def automatic_ocr_for_country(image, country):
    """Run only the OCR language associated with a known country."""
    language = COUNTRY_LANG_MAP.get((country or "").strip().lower(), "en")
    lines, _scores = extract_lines_with_scores(image, get_ocr_for_language(language))
    get_logger().info("OCR extracted text (%s): %s", language, lines)
    return lines

