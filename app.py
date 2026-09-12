import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify

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
    """Calculates cosine similarity between 1D source vector and 2D target matrix."""
    dot_products = np.dot(target_vecs, source_vec)
    source_norm = np.linalg.norm(source_vec)
    target_norms = np.linalg.norm(target_vecs, axis=1)
    
    denominator = source_norm * target_norms
    denominator[denominator == 0] = 1e-8  # Avoid division by zero
    return dot_products / denominator

def parse_items(text_field):
    """Converts a comma-separated string into a cleaned list of Title Case strings."""
    if not text_field:
        return []
    return [item.strip().title() for item in text_field.split(',') if item.strip()]

@app.route("/")

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/recipes.json")
def recipes_json():
    """Serve the recipe dataset so the frontend can build the dish-name autocomplete."""
    return jsonify(RECIPES)

@app.route("/search-text", methods=["POST"])
def search_text():
    dish_name = request.form.get("dish_name", "").strip()
    origin_country = request.form.get("origin_country", "").strip()
    target_country = request.form.get("target_country", "").strip()

    # 1. Locate source dish and record index
    source_dish = None
    source_idx = None
    for idx, recipe in enumerate(RECIPES):
        if (recipe["title"].lower() == dish_name.lower() and 
            recipe["country"].lower() == origin_country.lower()):
            source_dish = recipe.copy()
            source_idx = idx
            break

    if source_dish:
        if not source_dish.get("region"):
            source_dish["region"] = "No Specific"
    else:
        source_dish = {
            "title": dish_name,
            "country": origin_country,
            "region": "No Specific",
            "ingredient_text": "",
            "action_text": "",
            "directions": "No direct match found for this entry."
        }

    recommended_dish = None
    similarity_score = 0.0
    regional_variations = []
    international_similarities = []

    if source_idx is not None:
        # 2. Compute similarity score against ALL recipes in dataset
        total_scores = np.zeros(len(RECIPES))
        for feature_name, matrix in EMBEDDINGS.items():
            source_vec = matrix[source_idx]
            feature_sim = compute_cosine_similarity(source_vec, matrix)
            total_scores += WEIGHTS[feature_name] * feature_sim

        # 3. Target Country Matches: Rank and pick Main Dish + Regional Variations
        target_indices = [idx for idx, r in enumerate(RECIPES) if r["country"].lower() == target_country.lower()]

        if target_indices:
            # Sort target candidates by similarity score descending
            target_indices_sorted = sorted(target_indices, key=lambda i: total_scores[i], reverse=True)

            # Top 1 Target Dish (Main Dish)
            best_global_idx = target_indices_sorted[0]
            similarity_score = round(float(total_scores[best_global_idx]) * 100, 2)
            recommended_dish = RECIPES[best_global_idx].copy()
            if not recommended_dish.get("region"):
                recommended_dish["region"] = "No Specific"

            # Top 3 Regional Variations (excluding the main match)
            for idx in target_indices_sorted[1:4]:
                dish = RECIPES[idx].copy()
                dish["score"] = round(float(total_scores[idx]) * 100, 2)
                dish["region"] = dish.get("region") or "No Specific"
                regional_variations.append(dish)

        # 4. International Similarities: Top 5 dishes, one each from 5 distinct non-target countries
        all_indices_sorted = np.argsort(total_scores)[::-1]
        seen_countries = set([target_country.lower(), origin_country.lower()])

        for idx in all_indices_sorted:
            recipe = RECIPES[idx]
            country_lower = recipe["country"].lower()

            if country_lower not in seen_countries:
                seen_countries.add(country_lower)
                dish = recipe.copy()
                dish["score"] = round(float(total_scores[idx]) * 100, 2)
                dish["region"] = dish.get("region") or "No Specific"
                international_similarities.append(dish)

                if len(international_similarities) == 5:
                    break

    # --- ENTITY EXTRACTION LOGIC ---
    src_ings = set(parse_items(source_dish.get("ingredient_text", "")))
    rec_ings = set(parse_items(recommended_dish.get("ingredient_text", ""))) if recommended_dish else set()

    src_actions = set(parse_items(source_dish.get("action_text", "")))
    rec_actions = set(parse_items(recommended_dish.get("action_text", ""))) if recommended_dish else set()

    entities = {
        "ingredients": {
            "shared": sorted(list(src_ings & rec_ings)),
            "source_unique": sorted(list(src_ings - rec_ings)),
            "target_unique": sorted(list(rec_ings - src_ings))
        },
        "actions": {
            "shared": sorted(list(src_actions & rec_actions)),
            "source_unique": sorted(list(src_actions - rec_actions)),
            "target_unique": sorted(list(rec_actions - src_actions))
        }
    }

    return render_template(
        "results.html",
        source_dish=source_dish,
        recommended_dish=recommended_dish,
        target_country=target_country,
        entities=entities,
        similarity_score=similarity_score,
        regional_variations=regional_variations,
        international_similarities=international_similarities
    )

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/faqs")
def faqs():
    return render_template("faqs.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
