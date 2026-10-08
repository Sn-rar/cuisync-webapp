import difflib
import json
import os
import re
import threading
import unicodedata
from functools import lru_cache

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request
from werkzeug.middleware.proxy_fix import ProxyFix
import ocr
import softmatch as soft_settings
from softmatch import SoftMatcher
from explainer import init as init_explainer, build_evidence, get_explanation
from shapley import compute_shapley, influence_shares


app = Flask(__name__)
# Behind a Cloudflare Tunnel (run_public.py) requests arrive through a proxy. ProxyFix reads the
# X-Forwarded-* headers so Flask sees the real https link and visitor address. No effect locally.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
# Largest upload accepted (menu photos). Cloudflare itself rejects anything over 100 MB.
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# 1. Load JSON dataset from root directory
json_path = os.path.join(app.root_path, 'recipes.json')
with open(json_path, 'r', encoding='utf-8') as f:
    RECIPES = json.load(f)

# Safety check: recipes.json must be in the same order the .npy files were built from.
ORDER_FILE = os.path.join(app.root_path, 'embeddings_order.json')
if os.path.exists(ORDER_FILE):
    with open(ORDER_FILE, encoding='utf-8') as f:
        _order = json.load(f)
    if _order != [[r.get("title", ""), r.get("country", "")] for r in RECIPES]:
        raise RuntimeError("recipes.json does not match the .npy files. Re-run the vectorization.")

# 2. Item-level soft matching (softmatch.py): needs <feature>_item_vecs.npy and <feature>_items.json
#    for ingredients, actions, cookware and utensils, next to app.py.
_FEATURES = ("ingredients", "actions", "cookware", "utensils")
_missing = [
    name
    for feature in _FEATURES
    for name in (f"{feature}_item_vecs.npy", f"{feature}_items.json")
    if not os.path.exists(os.path.join(app.root_path, name))
]
if _missing:
    raise FileNotFoundError(
        "Missing data files next to app.py: " + ", ".join(_missing)
        + ". Download them from the Drive folder in the README (step 4)."
    )
MATCHER = SoftMatcher(app.root_path, RECIPES)

# On Render the app is started by gunicorn, so the __main__ block at the bottom
# never runs. Wake the OCR Space up in the background so the first upload is faster.
if ocr.OCR_API_URL and __name__ != "__main__":
    threading.Thread(target=ocr.warm_up, daemon=True).start()
if len(MATCHER) != len(RECIPES):
    raise RuntimeError(
        f"recipes.json has {len(RECIPES)} recipes but the exported files describe "
        f"{len(MATCHER)}. Use the recipes.json that was exported together with them."
    )


# 3. Define feature weights (Adjust ratios as needed to equal 1.0)
WEIGHTS = {
    "ingredients": 0.93,  
    "actions":     0.04,  
    "cookware":    0.02,  
    "utensils":    0.01  
}


# Minimum blended similarity (0.0 - 1.0) the best match must reach. Below this the website
# shows the "We Can't Find Your Dish" pop-up on the search page instead of an empty results page.
SIMILARITY_THRESHOLD = 0.35


def no_result_page(error_desc, error_title="We Can't Find Your Dish"):
    """Send the user back to the search page with the error pop-up."""
    return render_template("index.html", error_title=error_title, error_desc=error_desc)


def split_names(text):
    """Split "A, B; C (x, y) / D" into ["A", "B", "C (x, y)", "D"].
    The separators  , ; | /  and new lines only count outside brackets, so
    "Mie Goreng (fried noodles, spicy)" stays one name.
    Keep this in step with splitNames() in templates/index.html."""
    names, current, depth = [], [], 0
    for ch in str(text):
        if ch in "([":
            depth += 1
        elif ch in ")]" and depth > 0:
            depth -= 1
        if depth == 0 and ch in ",;|/\n\r":
            names.append("".join(current))
            current = []
        else:
            current.append(ch)
    names.append("".join(current))
    return [n.strip() for n in names if n.strip()]


def normalize_name(text):
    """Comparison form of a dish name: Unicode-normalised, lower-cased, single spaces."""
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def recipe_names(recipe):
    """Normalised title plus every alternative title of a recipe.
    An alternative_title that lists several names is accepted BOTH as the separate names
    and as the whole text, so the search works however the page sends it."""
    names = {normalize_name(recipe.get("title", ""))}
    raw = recipe.get("alternative_title") or recipe.get("alternative_titles") or []
    if not isinstance(raw, (list, tuple, set)):
        raw = [raw]
    for item in raw:
        if isinstance(item, str):
            names.add(normalize_name(item))
            names.update(normalize_name(name) for name in split_names(item))
    names.discard("")
    return names



def parse_items(text_field):
    if not text_field:
        return []
    return [item.strip().title() for item in text_field.split(",") if item.strip()]


# Let the explainer learn the dataset (typo fixes, generic cooking steps, grounding vocabulary)
init_explainer(RECIPES, parse_items)


def format_recipe_url(link):
    """Formats recipe link to ensure absolute HTTP/HTTPS URL standard."""
    if not link:
        return ""
    if link.startswith("http://") or link.startswith("https://"):
        return link
    return f"https://{link}"

@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")


# Finds the closest matching dish in the target country and compares ingredients, actions, and similar dishes.
def low_similarity_page(message, score, dish_title, origin_country, target_country, recipe_id=None):
    """Back to the search page with the low-similarity popup (View results anyway / No).
    The popup's 'View results anyway' button posts the same search again with force=1."""
    return render_template(
        "index.html",
        similarity_warning=message,
        similarity_score=round(score),
        warn_dish=dish_title,
        warn_origin=origin_country,
        warn_target=target_country,
        warn_recipe_id="" if recipe_id is None else recipe_id,
    )


def blend_scores(per_feature):
    """Weighted blend of the four per-feature score arrays (NaN = category empty in both dishes)."""
    weighted_sum = np.zeros(len(RECIPES))
    weight_used = np.zeros(len(RECIPES))
    for feature_name, weight in WEIGHTS.items():
        feature_scores = per_feature[feature_name]
        counted = ~np.isnan(feature_scores)
        weighted_sum += weight * np.where(counted, feature_scores, 0.0)
        weight_used += weight * counted
    return np.divide(weighted_sum, weight_used, out=np.zeros(len(RECIPES)), where=weight_used > 0)


def shapley_for(per_feature, idx):
    """Shapley values + influence shares + per-feature scores for the pair (source, recipe idx)."""
    raw_sims = {name: float(per_feature[name][idx]) for name in WEIGHTS}
    used = {name: not np.isnan(value) for name, value in raw_sims.items()}
    used_weight = sum(WEIGHTS[name] for name in WEIGHTS if used[name]) or 1.0
    effective_weights = {name: (WEIGHTS[name] / used_weight if used[name] else 0.0) for name in WEIGHTS}
    clean_sims = {name: (value if used[name] else 0.0) for name, value in raw_sims.items()}
    shapley = compute_shapley(clean_sims, effective_weights)
    return shapley, influence_shares(shapley), clean_sims


@lru_cache(maxsize=32)
def feature_scores_for_source(source_idx):
    """Per-feature scores of every recipe against one source dish (cached: pin clicks reuse it)."""
    return {name: MATCHER.scores(name, source_idx) for name in WEIGHTS}


def get_search_results(dish_name, origin_country, target_country, force=False, recipe_id=None):


    # ---- 2. find the source dish (exact title + origin country, ignoring capitals) ----
    source_dish = None
    source_idx = None
    # Several recipes can share a title or alternative title (e.g. five "Banh Mi" recipes in
    # Vietnam). When the dropdown sends the exact recipe_id, use that recipe, as long as it
    # really has this name and origin country. Otherwise fall back to the first name match.
    def is_match(recipe):
        return (
            normalize_name(dish_name) in recipe_names(recipe)
            and normalize_name(recipe["country"]) == normalize_name(origin_country)
        )

    if recipe_id not in (None, ""):
        for index, recipe in enumerate(RECIPES):
            if str(recipe.get("recipe_id")) == str(recipe_id).strip() and is_match(recipe):
                source_dish = recipe.copy()
                source_idx = index
                break

    if source_dish is None:
        for index, recipe in enumerate(RECIPES):
            if is_match(recipe):
                source_dish = recipe.copy()
                source_idx = index
                break

    if source_dish is None:
        same_title = sorted({r["country"] for r in RECIPES if normalize_name(dish_name) in recipe_names(r)})
        close = difflib.get_close_matches(dish_name.lower(), [r["title"].lower() for r in RECIPES], n=5, cutoff=0.6)
        if same_title:
            return no_result_page(
                f'"{dish_name}" is in our dataset, but under {", ".join(c.title() for c in same_title)} '
                f"rather than {origin_country.title()}."
            )
        return no_result_page(
            f'"{dish_name}" from {origin_country.title()} is not in our dataset, '
            "so there is nothing to compare."
        )


    if not source_dish.get("region"):
        source_dish["region"] = "No Specific"
    source_dish["recipe_link"] = format_recipe_url(source_dish.get("recipe_link", ""))
    source_dish["idx"] = source_idx

    recommended_dish = None
    similarity_score = 0.0
    regional_variations = []
    international_similarities = []

    # ---- 3. score every recipe against the source dish ----
    per_feature = feature_scores_for_source(source_idx)
    total_scores = blend_scores(per_feature)

    # The searched dish itself is never a candidate. Without this, a search where origin and
    # target are the same country matches the dish with itself (100%), so every entity shows
    # as "shared" and no GPT explanation is generated.
    target_indices = [
        index
        for index, recipe in enumerate(RECIPES)
        if recipe["country"].lower() == target_country.lower() and index != source_idx
    ]
    if not target_indices:
        if origin_country.lower() == target_country.lower():
            return no_result_page(
                f'Our dataset has no other recipes from {target_country.title()} '
                f'to compare with "{source_dish["title"]}".'
            )
        return no_result_page(
            f"Our dataset has no recipes from {target_country.title()}."
        )

    ordered_targets = sorted(
        target_indices, key=lambda index: total_scores[index], reverse=True
    )
    best_index = ordered_targets[0]



    # Below the threshold: do NOT load the results yet. Go back to the search page and let the
    # user choose (popup: "View results anyway" re-posts with force=1, "No" stays on the page).
    if total_scores[best_index] < SIMILARITY_THRESHOLD and not force:
        return low_similarity_page(
            f'No dish from {target_country.title()} in our dataset is similar enough to '
            f'"{source_dish["title"]}" to recommend (closest match is only '
            f"{total_scores[best_index] * 100:.0f}% similar).",
            float(total_scores[best_index]) * 100,
            source_dish["title"],
            source_dish["country"],
            target_country,
            source_dish.get("recipe_id"),
        )

    similarity_score = round(float(total_scores[best_index]) * 100, 2)

    # Shapley values of the four entity-type scores (Chapter 3 formula). Used only as input to
    # the GPT explanation. A category that is empty in both dishes (NaN) is left out and the
    # other weights are re-normalised exactly like total_scores, so the values still add up
    # to the displayed score.
    shapley, influence, clean_sims = shapley_for(per_feature, best_index)
    feature_scores = {name: round(value, 4) for name, value in clean_sims.items()}
    if abs(sum(shapley.values()) * 100 - similarity_score) > 0.01:
        app.logger.warning("Shapley values do not add up to the similarity score")
    recommended_dish = RECIPES[best_index].copy()
    recommended_dish["idx"] = int(best_index)
    if not recommended_dish.get("region"):
        recommended_dish["region"] = "No Specific"
    recommended_dish["recipe_link"] = format_recipe_url(
        recommended_dish.get("recipe_link", "")
    )

    for idx in ordered_targets[1:4]:
        dish = RECIPES[idx].copy()
        dish["idx"] = int(idx)
        dish["score"] = round(float(total_scores[idx]) * 100, 2)
        dish["region"] = dish.get("region") or "No Specific"
        dish["recipe_link"] = format_recipe_url(dish.get("recipe_link", ""))
        regional_variations.append(dish)

    seen_countries = {target_country.lower(), origin_country.lower()}
    for index in np.argsort(total_scores)[::-1]:
        recipe = RECIPES[index]
        recipe_country = recipe["country"].lower()
        if recipe_country in seen_countries:
            continue
        seen_countries.add(recipe_country)
        dish = recipe.copy()
        dish["idx"] = int(index)
        dish["score"] = round(float(total_scores[index]) * 100, 2)
        dish["region"] = dish.get("region") or "No Specific"
        dish["recipe_link"] = format_recipe_url(dish.get("recipe_link", ""))
        international_similarities.append(dish)
        if len(international_similarities) == 5:
            break


    # --- ENTITY EXTRACTION & COMPARISON LOGIC ---
    source_ingredients = set(parse_items(source_dish.get("ingredient_text", "")))
    recommended_ingredients = (
        set(parse_items(recommended_dish.get("ingredient_text", "")))
        if recommended_dish
        else set()
    )
    source_actions = set(parse_items(source_dish.get("action_text", "")))
    recommended_actions = (
        set(parse_items(recommended_dish.get("action_text", "")))
        if recommended_dish
        else set()
    )
    source_cookware = set(parse_items(source_dish.get("cookware_text", "")))
    recommended_cookware = (
        set(parse_items(recommended_dish.get("cookware_text", "")))
        if recommended_dish
        else set()
    )
    source_utensils = set(parse_items(source_dish.get("utensil_text", "")))
    recommended_utensils = (
        set(parse_items(recommended_dish.get("utensil_text", "")))
        if recommended_dish
        else set()
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
        "cookware": {
            "shared": sorted(source_cookware & recommended_cookware),
            "source_unique": sorted(source_cookware - recommended_cookware),
            "target_unique": sorted(recommended_cookware - source_cookware),
        },
        "utensils": {
            "shared": sorted(source_utensils & recommended_utensils),
            "source_unique": sorted(source_utensils - recommended_utensils),
            "target_unique": sorted(recommended_utensils - source_utensils),
        },
    }

    # --- AI EXPLANATION (GPT-4o via explainer.py; cached; template fallback) ---
    explanation = None
    explanation_source = None
    if best_index != source_idx:
        evidence = build_evidence(source_dish, recommended_dish, similarity_score, influence)
        explanation, explanation_source = get_explanation(evidence)

    # Weight percentages mapped for UI progress bars
    feature_weights = {
        key: round(value * 100) for key, value in WEIGHTS.items()
    }

    return render_template(
        "results.html",
        source_dish=source_dish,
        recommended_dish=recommended_dish,
        target_country=target_country,
        entities=entities,
        similarity_score=similarity_score,
        feature_weights=feature_weights,
        regional_variations=regional_variations,
        international_similarities=international_similarities,
        explanation=explanation,
        explanation_source=explanation_source,
        feature_scores=feature_scores,
        shap_values=shapley,
        influence=influence or {},
    )


@app.route("/api/explain-pin")
def explain_pin():
    """
    GPT explanation for ONE dish on the map (regional variation / international similarity),
    compared with the searched dish. Same pipeline as the main match: cleaned evidence ->
    cache -> GPT-4o -> grounding check. The scores are recomputed here from the recipe
    indices, so the browser cannot influence what is sent to GPT.
    """
    try:
        source_idx = int(request.args.get("source", ""))
        idx = int(request.args.get("idx", ""))
    except ValueError:
        return jsonify({"error": "source and idx must be integers"}), 400
    if not (0 <= source_idx < len(RECIPES) and 0 <= idx < len(RECIPES)) or source_idx == idx:
        return jsonify({"error": "invalid recipe index"}), 400

    per_feature = feature_scores_for_source(source_idx)
    score = round(float(blend_scores(per_feature)[idx]) * 100, 2)
    _, influence, clean_sims = shapley_for(per_feature, idx)

    source_dish, dish = RECIPES[source_idx], RECIPES[idx]
    evidence = build_evidence(source_dish, dish, score, influence)
    explanation, explanation_source = get_explanation(evidence)
    return jsonify({
        "idx": idx,
        "explanation": explanation,
        "source": explanation_source,          # "ai" = GPT, "template" = fallback
        "featureScores": {k: round(v, 4) for k, v in clean_sims.items()},
    })


@app.errorhandler(413)
def file_too_large(_error):
    """Upload bigger than MAX_CONTENT_LENGTH."""
    message = "This photo is too large. Please upload an image under 16 MB."
    if request.path == "/extract-dish-info":
        return jsonify({"success": False, "error": "too_large", "message": message, "matches": []}), 413
    return render_template("index.html", error_title="Image Too Large", error_desc=message), 413


@app.route("/recipes.json")
def recipes_json():
    return jsonify(RECIPES)


@app.route("/search-text", methods=["POST"])
def search_text():
    return get_search_results(
        request.form.get("dish_name", "").strip(),
        request.form.get("origin_country", "").strip(),
        request.form.get("target_country", "").strip(),
        force=request.form.get("force") == "1",
        recipe_id=request.form.get("recipe_id", "").strip() or None,
    )


@app.route("/extract-dish-info", methods=["POST"])
def extract_dish_info():
    if "dish_image" not in request.files:
        return jsonify({"success": False, "message": "No image uploaded"})

    try:
        raw = request.files["dish_image"].read()
        image_bytes = np.frombuffer(raw, np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"success": False, "message": "Invalid image format"})

        try:
            lines = ocr.automatic_ocr(image)
        except ocr.UnsupportedScriptError as error:
            # Khmer / Burmese / unreadable. If some dish still matched (e.g. an
            # English line on the menu), carry on, otherwise tell the user.
            lines = error.lines
            if not ocr.find_recipes(ocr.romanize_lines(lines), recipes=RECIPES):
                return jsonify({
                    "success": False,
                    "error": "unsupported_script",
                    "message": str(error),
                    "raw_text": "\n".join(lines),
                    "romanized_text": "",
                    "matches": [],
                })

        if not lines:
            return jsonify({
                "success": False,
                "error": "no_text",
                "message": "We couldn't find any text in this photo or just try it again. Make sure the menu is in focus and the dish names are visible.",
                "raw_text": "(No readable text detected)",
                "romanized_text": "(None)",
                "matches": [],
            })

        romanized = ocr.romanize_lines(lines)
        matches = ocr.find_recipes(romanized, recipes=RECIPES)
        return jsonify({
            "success": bool(matches),
            "message": "" if matches else "No matching recipe found in dataset.",
            "raw_text": "\n".join(lines),
            "romanized_text": "\n".join(romanized),
            "matches": matches,
        })
    except Exception as error:
        return jsonify({"success": False, "message": f"Processing error: {error}"})


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

        try:
            lines = ocr.automatic_ocr(image)
            unsupported_error = None
        except ocr.UnsupportedScriptError as error:
            lines = error.lines
            unsupported_error = error

        matches = ocr.find_recipes(ocr.romanize_lines(lines), recipes=RECIPES)
        if not matches and unsupported_error:
            return render_template("index.html", error_title="Language Not Supported", error_desc=str(unsupported_error))
        if not matches:
            return render_template("index.html", error_title="No Recipe Record", error_desc="The uploaded image does not contain a recipe name found in our dataset.")
        recipe = matches[0]
        return get_search_results(recipe["title"], recipe["country"], target_country,
                                  recipe_id=recipe.get("recipe_id"))
    except Exception as error:
        return render_template("index.html", error_title="Processing Error", error_desc=f"An error occurred while processing the image: {error}")


@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

#---ANALYSIS PAGE API---

COUNTRY_REGIONS = {
    "CHINA": "East Asia",
    "JAPAN": "East Asia",
    "SOUTH KOREA": "East Asia",

    "CAMBODIA": "Southeast Asia",
    "INDONESIA": "Southeast Asia",
    "MALAYSIA": "Southeast Asia",
    "MYANMAR": "Southeast Asia",
    "PHILIPPINES": "Southeast Asia",
    "THAILAND": "Southeast Asia",
    "VIETNAM": "Southeast Asia",

    "INDIA": "South Asia",

    "SAUDI ARABIA": "West Asia",
    "TURKIYE": "West Asia"
}

#ANALYSIS PAGE: 1. Normalize the country name and grouped them into regions.
def normalize_analysis_country(country):
    if not country:
        return ""

    return " ".join(
        str(country).strip().upper().split()
    )


def get_analysis_region(country):
    return COUNTRY_REGIONS.get(
        normalize_analysis_country(country),
        "Other Asia"
    )

def get_analysis_countries():
    regions = {}

    for recipe in RECIPES:
        country = normalize_analysis_country(
            recipe.get("country")
        )

        if not country:
            continue

        region = get_analysis_region(country)

        if region not in regions:
            regions[region] = set()

        regions[region].add(country)

    return {
        region: sorted(countries)
        for region, countries in sorted(regions.items())
    }

#ANALYSIS PAGE: 2. Function to get the recipe count per country.

def get_analysis_country_counts():
    counts = {}

    for recipe in RECIPES:
        country = normalize_analysis_country(
            recipe.get("country")
        )

        if not country:
            continue

        counts[country] = counts.get(country, 0) + 1

    return counts

#ANALYSIS PAGE: 3. Function to get the recipe count per region
def get_analysis_region_counts():
    counts = {}

    for recipe in RECIPES:
        country = normalize_analysis_country (recipe.get("country"))

        if not country:
            continue

        region = get_analysis_region(country)
        counts[region] = counts.get(region,0) + 1

    return counts

#ANALYSIS PAGE: 4. Returns the count of the entity, which sorts everything from most common to least common.
def get_analysis_top_entities(recipes, field, limit=15):
    total_recipes = len(recipes)

    if total_recipes == 0:
        return []

    entity_recipe_counts = {}

    for recipe in recipes:
        value = recipe.get(field, "")

        if not value:
            continue

        # Count each entity only once per recipe.
        items = set(parse_items(value))

        for item in items:
            if item:
                entity_recipe_counts[item] = (
                    entity_recipe_counts.get(item, 0) + 1
                )

    results = []

    for entity, count in entity_recipe_counts.items():
        percentage = (count / total_recipes) * 100

        results.append({
            "name": entity,
            "count": count,
            "percentage": round(percentage, 2)
        })

    results.sort(
        key=lambda x: x["count"],
        reverse=True
    )

    return results[:limit]

#ANALYSIS PAGE: 4. Function to get analysis on the country selected by the user.
def get_analysis_recipes_for_countries(countries):
    # If there are not selected countries, all are queried.
    if not countries:
        return RECIPES

    selected_countries = {
        normalize_analysis_country(country)
        for country in countries
        if normalize_analysis_country(country)
    }

    if not selected_countries:
        return RECIPES

    return [
        recipe
        for recipe in RECIPES
        if normalize_analysis_country(
            recipe.get("country")
        ) in selected_countries
    ]

#ANALYSIS PAGE: 4. Function to get the overview analysis of the recipe count per category.
@app.route("/api/analysis/overview")
def analysis_overview():
    return jsonify({
        "countries_by_region": #Calls the helper function to count the recipe per region.
            get_analysis_countries(), 

        "country_counts":
            get_analysis_country_counts(), #Calls the helper function to count the recipe per country.

        "region_counts":
            get_analysis_region_counts(), #Calls the helper function to count the recipe per region.

        "total_recipes": #Count the number of recipes on the dataset.
            len(RECIPES)
    })

#ANALYSIS PAGE: 5. Function that receives the user's selected country and return the top ingredients results back.
@app.route("/api/analysis/ingredients")
def analysis_ingredients():
    countries = request.args.getlist("country")

    selected_recipes = (
        get_analysis_recipes_for_countries(countries)
    )

    return jsonify({
        "countries": countries,
        "selected_country_count": len(countries),
        "recipe_count": len(selected_recipes),
        "results": get_analysis_top_entities(
            selected_recipes,
            "ingredient_text",
            limit=15
        )
    })

#ANALYSIS PAGE: 5. Function that receives the user's selected country and return the top directions results back.
@app.route("/api/analysis/instructions")
def analysis_instructions():
    countries = request.args.getlist("country")

    instruction_type = request.args.get(
        "type",
        "actions"
    ).strip().lower()

    field_map = {
        "actions": "action_text",
        "cooking_actions": "action_text",
        "utensils": "utensil_text",
        "cookware": "cookware_text"
    }

    field = field_map.get(
        instruction_type,
        "action_text"
    )

    selected_recipes = (
        get_analysis_recipes_for_countries(countries)
    )

    return jsonify({
        "countries": countries,
        "selected_country_count": len(countries),
        "type": instruction_type,
        "recipe_count": len(selected_recipes),
        "results": get_analysis_top_entities(
            selected_recipes,
            field,
            limit=10
        )
    })

@app.route("/faqs")
def faqs():
    return render_template("faqs.html")


if __name__ == "__main__":
    # Load the OCR models in the background so the first upload is fast.
    # With the reloader off there is only one process, so start the warm-up directly.
    # If the reloader is ever turned on, Flask starts the app twice (a watcher + the
    # real app); WERKZEUG_RUN_MAIN is only set in the real one, so load them once.
    USE_RELOADER = False
    if not USE_RELOADER or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Thread(target=ocr.warm_up, daemon=True).start()

    app.run(host="127.0.0.1", port=8000, debug=False, use_reloader=USE_RELOADER)