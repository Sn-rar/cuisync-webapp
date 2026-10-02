import json
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
import ocr


app = Flask(__name__)


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
        if not source_dish.get("region"):
            source_dish["region"] = "No Specific"
        source_dish["recipe_link"] = format_recipe_url(source_dish.get("recipe_link", ""))
    else:
        source_dish = {
            "title": dish_name,
            "country": origin_country,
            "country_region": "",
            "region": "No Specific",
            "ingredient_text": "",
            "action_text": "",
            "cookware_text": "",
            "utensil_text": "",
            "directions": "No direct match found for this entry.",
            "recipe_link": "",
            "image_link": "",
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
            if not recommended_dish.get("region"):
                recommended_dish["region"] = "No Specific"
            recommended_dish["recipe_link"] = format_recipe_url(
                recommended_dish.get("recipe_link", "")
            )

            for idx in ordered_targets[1:4]:
                dish = RECIPES[idx].copy()
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
    )


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

        lines = ocr.automatic_ocr(image)
        if not lines:
            return jsonify({
                "success": False,
                "message": "No text detected in image.",
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

        lines = ocr.automatic_ocr(image)
        matches = ocr.find_recipes(ocr.romanize_lines(lines), recipes=RECIPES)
        if not matches:
            return render_template("index.html", error_title="No Recipe Record", error_desc="The uploaded image does not contain a recipe name found in our dataset.")
        recipe = matches[0]
        return get_search_results(recipe["title"], recipe["country"], target_country)
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
    app.run(host="127.0.0.1", port=8000, debug=True)