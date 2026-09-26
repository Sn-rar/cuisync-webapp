import json
import os
import re
import sys
from difflib import SequenceMatcher

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from paddleocr import PaddleOCR

# Normal deployments install Unidecode from requirements.txt. The local vendor
# folder lets this copy run as well when the active Python environment lacks it.
try:
    from unidecode import unidecode
except ModuleNotFoundError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".vendor"))
    from unidecode import unidecode


app = Flask(__name__)

# PaddleOCR language codes. Khmer and Burmese are not covered by the standard
# Latin recognizer; use a dedicated recognition model for those scripts.
COUNTRY_LANG_MAP = {
    "china": "ch",
    "india": "hi",
    "indonesia": "id",
    "japan": "japan",
    "malaysia": "ms",
    "philippines": "tl",
    "saudi arabia": "ar",
    "south korea": "korean",
    "thailand": "th",
    "turkiye": "tr",
    "turkey": "tr",
    "vietnam": "vi",
}

OCR_ENGINES = {}
AUTO_OCR_LANGUAGES = ("en", *sorted(set(COUNTRY_LANG_MAP.values())))


def get_ocr_for_language(lang):
    """Load and cache a PaddleOCR engine for one supported language."""
    if lang not in OCR_ENGINES:
        OCR_ENGINES[lang] = PaddleOCR(
            lang=lang,
            ocr_version="PP-OCRv5",
            # Skip costly document-layout steps for the usual upright menu photo.
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # A menu normally needs only a 960 px detection pass. This is much
            # quicker on phones than processing the original multi-megapixel
            # upload, while retaining readable menu-sized text.
            text_det_limit_side_len=960,
            text_det_limit_type="max",
            text_recognition_batch_size=16,
        )
    return OCR_ENGINES[lang]


def get_ocr(country=""):
    """Load the OCR engine mapped to a country, defaulting to English."""
    lang = COUNTRY_LANG_MAP.get((country or "").strip().lower(), "en")
    return get_ocr_for_language(lang)


# 1. Load JSON dataset from root directory
json_path = os.path.join(app.root_path, 'recipes.json')
with open(json_path, 'r', encoding='utf-8') as f:
    RECIPES = json.load(f)

# 2. Load 4 separate .npy feature embedding matrices
EMBEDDINGS = {
    "ingredients": np.load(os.path.join(app.root_path, 'ingredients.npy')),
    "actions":     np.load(os.path.join(app.root_path, 'actions.npy')),
    "cookware":    np.load(os.path.join(app.root_path, 'cookware.npy')),
    "utensils":    np.load(os.path.join(app.root_path, 'utensils.npy'))
}

# 3. Define feature weights (Adjust ratios as needed to equal 1.0)
WEIGHTS = {
    "ingredients": 0.93,  
    "actions":     0.04,  
    "cookware":    0.02,  
    "utensils":    0.01  
}


def compute_cosine_similarity(source_vec, target_vecs):
    dot_products = np.dot(target_vecs, source_vec)
    source_norm = np.linalg.norm(source_vec)
    target_norms = np.linalg.norm(target_vecs, axis=1)
    denominator = source_norm * target_norms
    denominator[denominator == 0] = 1e-8
    return dot_products / denominator


def parse_items(text_field):
    if not text_field:
        return []
    return [item.strip().title() for item in text_field.split(",") if item.strip()]





# Finds the closest matching dish in the target country and compares ingredients, actions, and similar dishes.
def get_search_results(dish_name, origin_country, target_country):
    source_dish = None
    source_idx = None

    for index, recipe in enumerate(RECIPES):
        if (
            recipe["title"].lower() == dish_name.lower()
            and recipe["country"].lower() == origin_country.lower()
        ):
            source_dish = recipe.copy()
            source_idx = index
            break

    if source_dish:
        source_dish["region"] = source_dish.get("region") or "No Specific"
    else:
        source_dish = {
            "title": dish_name,
            "country": origin_country,
            "region": "No Specific",
            "ingredient_text": "",
            "action_text": "",
            "directions": "No direct match found for this entry.",
        }

    recommended_dish = None
    similarity_score = 0.0
    regional_variations = []
    international_similarities = []

    if source_idx is not None:
        total_scores = np.zeros(len(RECIPES))
        for feature_name, matrix in EMBEDDINGS.items():
            if len(matrix) > source_idx:
                total_scores += WEIGHTS[feature_name] * compute_cosine_similarity(
                    matrix[source_idx], matrix
                )

        target_indices = [
            index
            for index, recipe in enumerate(RECIPES)
            if recipe["country"].lower() == target_country.lower()
        ]
        if target_indices:
            ordered_targets = sorted(
                target_indices, key=lambda index: total_scores[index], reverse=True
            )
            best_index = ordered_targets[0]
            similarity_score = round(float(total_scores[best_index]) * 100, 2)
            recommended_dish = RECIPES[best_index].copy()
            recommended_dish["region"] = (
                recommended_dish.get("region") or "No Specific"
            )

            for index in ordered_targets[1:4]:
                dish = RECIPES[index].copy()
                dish["score"] = round(float(total_scores[index]) * 100, 2)
                dish["region"] = dish.get("region") or "No Specific"
                regional_variations.append(dish)

        seen_countries = {target_country.lower(), origin_country.lower()}
        for index in np.argsort(total_scores)[::-1]:
            recipe = RECIPES[index]
            recipe_country = recipe["country"].lower()
            if recipe_country in seen_countries:
                continue
            seen_countries.add(recipe_country)
            dish = recipe.copy()
            dish["score"] = round(float(total_scores[index]) * 100, 2)
            dish["region"] = dish.get("region") or "No Specific"
            international_similarities.append(dish)
            if len(international_similarities) == 5:
                break

    source_ingredients = set(parse_items(source_dish.get("ingredient_text", "")))
    recommended_ingredients = set(
        parse_items(recommended_dish.get("ingredient_text", ""))
        if recommended_dish
        else []
    )
    source_actions = set(parse_items(source_dish.get("action_text", "")))
    recommended_actions = set(
        parse_items(recommended_dish.get("action_text", ""))
        if recommended_dish
        else []
    )

    entities = {
        "ingredients": {
            "shared": sorted(source_ingredients & recommended_ingredients),
            "source_unique": sorted(source_ingredients - recommended_ingredients),
            "target_unique": sorted(recommended_ingredients - source_ingredients),
        },
        "actions": {
            "shared": sorted(source_actions & recommended_actions),
            "source_unique": sorted(source_actions - recommended_actions),
            "target_unique": sorted(recommended_actions - source_actions),
        },
    }

    return render_template(
        "results.html",
        source_dish=source_dish,
        recommended_dish=recommended_dish,
        target_country=target_country,
        entities=entities,
        similarity_score=similarity_score,
        regional_variations=regional_variations,
        international_similarities=international_similarities,
    )

# Converts a PaddleOCR result into usable dictionary data.
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


def ocr_quality(lines, scores):
    """Prefer confident results that recognize more meaningful text."""
    return sum(max(score, 0.0) * len(line) for line, score in zip(lines, scores))


def automatic_ocr(image):
    """Detect the best supported OCR language without requiring a country choice."""
    best_lines = []
    best_score = -1.0
    best_language = "en"

    for language in AUTO_OCR_LANGUAGES:
        try:
            lines, scores = extract_lines_with_scores(
                image, get_ocr_for_language(language)
            )
        except Exception as error:
            print(f"OCR model '{language}' skipped: {error}", flush=True)
            continue
        score = ocr_quality(lines, scores)
        if score > best_score:
            best_lines = lines
            best_score = score
            best_language = language

    print(f"\n--- OCR EXTRACTED TEXT ({best_language}) ---", flush=True)
    for line in best_lines:
        print(line, flush=True)
    print("-----------------------------------\n", flush=True)
    return best_lines

# Converts non-Latin text into Latin characters without translating it.
def romanize_text(text):
    """Convert non-Latin writing to Latin characters without translating it."""
    return re.sub(r"\s+", " ", unidecode(str(text))).strip()

def romanize_lines(lines):
    """Romanize OCR text and display it as one sentence in the terminal."""
    romanized = [romanize_text(line) for line in lines if str(line).strip()]

    print("\n--- ROMANIZED TEXT ---", flush=True)
    print(" ".join(romanized), flush=True)
    print("-----------------------\n", flush=True)

    return romanized

# Creates a simplified text key for matching by removing accents, spaces, and punctuation.
def match_key(text):
    """Create a punctuation- and accent-insensitive key for recipe matching."""
    return re.sub(r"[^a-z0-9]+", "", romanize_text(text).casefold())

# Builds possible dish names by combining nearby OCR text lines.
def menu_candidates(lines, romanized_lines):
    """Build complete dish candidates from OCR fragments on neighboring lines.

    OCR commonly separates a wrapped menu title into two or more recognition
    boxes. Matching only a single box loses those dishes, so we also test
    short consecutive windows and the complete menu reading order.
    """
    candidates = set()
    for source_lines in (lines, romanized_lines):
        keys = [match_key(line) for line in source_lines if match_key(line)]
        candidates.update(keys)

        # PaddleOCR can return every word as its own box, particularly for a
        # chalkboard or handwritten menu. Ten boxes cover a long wrapped name
        # without combining a whole page into every candidate.
        for start in range(len(keys)):
            joined = ""
            for end in range(start, min(start + 10, len(keys))):
                joined += keys[end]
                candidates.add(joined)

        if keys:
            candidates.add("".join(keys))
    return candidates

# Checks whether a recipe title matches an OCR candidate, allowing for small OCR mistakes.
def is_recipe_match(title_key, candidates):
    """Match exact menu text, tolerating a small OCR typo in a full title."""
    for candidate in candidates:
        if title_key in candidate:
            return True

        # Only compare similarly sized strings. This catches errors such as
        # "Turkisn" for "Turkish" without treating a single word as a dish.
        if (
            len(title_key) >= 8
            and abs(len(title_key) - len(candidate)) <= 4
            and SequenceMatcher(None, title_key, candidate).ratio() >= 0.84
        ):
            return True
    return False

# Finds all recipes recognized from menu text, optionally filtered by country.
def find_recipes(lines, romanized_lines, origin_country=""):
    """Return every database recipe recognized in a menu, without translating."""
    candidate_keys = menu_candidates(lines, romanized_lines)
    origin_key = origin_country.strip().casefold()
    matches = []
    seen = set()

    for recipe in RECIPES:
        title = recipe.get("title", "").strip()
        country = recipe.get("country", "").strip()
        if not title or (origin_key and country.casefold() != origin_key):
            continue

        title_key = match_key(title)
        if is_recipe_match(title_key, candidate_keys):
            identity = (title.casefold(), country.casefold())
            if identity not in seen:
                matches.append({"title": title, "country": country})
                seen.add(identity)

    # Prefer a specific detected name over a shorter database title contained
    # inside it (for example, "Ambyachi Kheer" over the separate "Kheer").
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

@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")


@app.route("/recipes.json")
def recipes_json():
    return jsonify(RECIPES)


@app.route("/search-text", methods=["POST"])

def search_text():
    return get_search_results(
        request.form.get("dish_name", "").strip(),
        request.form.get("origin_country", "").strip(),
        request.form.get("target_country", "").strip(),
    )


@app.route("/extract-dish-info", methods=["POST"])
def extract_dish_info():
    if "dish_image" not in request.files:
        return jsonify({"success": False, "message": "No image uploaded"})

    try:
        image_bytes = np.frombuffer(request.files["dish_image"].read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"success": False, "message": "Invalid image format"})

        lines = automatic_ocr(image)
        if not lines:
            return jsonify({
                "success": False,
                "message": "No text detected in image.",
                "raw_text": "(No readable text detected)",
                "romanized_text": "(None)",
                "matches": [],
            })

        romanized = romanize_lines(lines)
        # `origin_country` is only an OCR language hint. Never use a previous
        # selection to filter a newly uploaded menu: infer every match's
        # country from the recipe database instead.
        matches = find_recipes(lines, romanized)
        return jsonify({
            "success": bool(matches),
            "message": "" if matches else "No matching recipe found in dataset.",
            "raw_text": "\n".join(lines),
            "romanized_text": "\n".join(romanized),
            "matches": matches,
        })
    except Exception as error:
        return jsonify({"success": False, "message": f"Processing error: {error}"})




#================== 
#                   OCR PART 
#                       ==============================

@app.route("/search-image", methods=["POST"])
def search_image():
    origin_country = request.form.get("origin_country", "").strip()
    target_country = request.form.get("target_country", "").strip()
    extracted_dish_name = request.form.get("extracted_dish_name", "").strip()

    try:
        # A chosen menu item no longer needs the original file. This keeps the
        # change-dish flow working after the browser returns from results and
        # clears its file input for security.
        if extracted_dish_name and origin_country:
            return get_search_results(extracted_dish_name, origin_country, target_country)

        if "dish_image" not in request.files:
            return render_template("index.html", error_title="No Recipe Record", error_desc="No image file was uploaded.")

        file = request.files["dish_image"]
        if not file.filename:
            return render_template("index.html", error_title="No Recipe Record", error_desc="No selected image file.")

        image_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return render_template("index.html", error_title="No Recipe Record", error_desc="Image format is not supported or corrupted.")

        lines = automatic_ocr(image)
        matches = find_recipes(lines, romanize_lines(lines))
        if not matches:
            return render_template("index.html", error_title="No Recipe Record", error_desc="The uploaded image does not contain a recipe name found in our dataset.")
        recipe = matches[0]
        return get_search_results(recipe["title"], recipe["country"], target_country)
    except Exception as error:
        return render_template("index.html", error_title="Processing Error", error_desc=f"An error occurred while processing the image: {error}")


@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


@app.route("/faqs")
def faqs():
    return render_template("faqs.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
