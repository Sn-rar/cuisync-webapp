"""Soft ingredient matching for Cuisync (put this file next to app.py).

For two recipes A and B, every ingredient in A is matched to its CLOSEST ingredient in B
(by language-model similarity) and vice-versa, so 'chicken breast' ~ 'chicken thigh' and
'ginger' ~ 'ginger root' count as near-matches, while 'chicken' vs 'pork' counts for little.
"""
import json
import os
import re

import numpy as np

FEATURES = ("ingredients", "actions", "cookware", "utensils")

# Similarities at or below FLOOR count as 0 (unrelated); the rest are stretched to 0..1.
# Raise it if unrelated items still score too well, lower it if close variants score too low.
# Use the "Pair similarity" printout from vectorize_softmatch.py to choose.
FLOOR = 0.35

# 0 = every item counts equally. If your lists are written main-ingredient-first, try 0.15-0.3:
# item at position k gets weight 1 / (1 + POSITION_DECAY * k), so early items matter more.
POSITION_DECAY = 0.0

# Items that appear in many recipes (salt, oil, water, garlic...) say little about a dish; rare,
# defining items (chicken, tilapia, lemongrass...) say a lot. IDF_POWER scales that effect:
# 0 = ignore, 1 = standard, 2 = even stronger. This is what stops shared batter/seasoning
# ingredients from outweighing a mismatched main ingredient.
IDF_POWER = 1.0

# How much more "is the source dish's makeup covered by the target?" (recall) counts than
# "is everything in the target found in the source?" (precision). 1 = equal (plain F1).
# With 2-3, extra ingredients in the target are forgiven more, which suits "find the closest
# dish to X" - a dish with a few extra ingredients isn't penalised as hard as one that is
# missing X's main ingredient.
RECALL_WEIGHT = 2.0

# An ingredient whose word also appears in the dish TITLE ("chicken" in "Fried Chicken") is the
# dish's headline ingredient, so it counts this many times more. 1 = off. Only used for
# ingredients, and only when the recipes are passed to SoftMatcher (app.py does this).
TITLE_BOOST = 4.0

_STOP = {"and", "with", "the", "of", "in", "a", "style", "or"}


def _words(text):
    """Lower-case words with a crude plural strip: 'Eggs' -> 'egg'."""
    out = set()
    for w in re.findall(r"[a-z]+", (text or "").lower()):
        if w in _STOP or len(w) < 3:
            continue
        out.add(w[:-1] if w.endswith("s") and len(w) > 3 else w)
    return out


def _weights(item_ids, idf, boost):
    """Normalised weight per item: rarer, earlier and title-named items count more."""
    position = 1.0 / (1.0 + POSITION_DECAY * np.arange(len(item_ids)))
    w = position * idf[item_ids] ** IDF_POWER * boost
    return w / w.sum()


class SoftMatcher:
    def __init__(self, folder, recipes=None):
        self.vecs, self.recipe_ids, self.idf, self.items, self.boost = {}, {}, {}, {}, {}
        for feature in FEATURES:
            self.vecs[feature] = np.load(os.path.join(folder, f"{feature}_item_vecs.npy"))
            with open(os.path.join(folder, f"{feature}_items.json"), encoding="utf-8") as f:
                data = json.load(f)
            self.recipe_ids[feature] = [np.array(ids, dtype=np.int64) for ids in data["recipes"]]
            if "df" in data:                                   # written by vectorize_softmatch.py
                df = np.array(data["df"], dtype=np.float64)
            else:                                              # written by colab_export_cell.py
                df = np.zeros(len(data["items"]))
                for ids in data["recipes"]:
                    for i in set(ids):
                        df[i] += 1
            n_recipes = data.get("n_recipes", len(data["recipes"]))
            self.idf[feature] = np.log((n_recipes + 1.0) / (df + 1.0)) + 1.0
            self.items[feature] = data["items"]
            self.boost[feature] = [np.ones(len(ids)) for ids in self.recipe_ids[feature]]

        # Headline-ingredient boost from the dish title (ingredients only).
        if recipes is not None and TITLE_BOOST != 1.0:
            names = self.items["ingredients"]
            for r, (recipe, ids) in enumerate(zip(recipes, self.recipe_ids["ingredients"])):
                title_words = _words(recipe.get("title", "")) | _words(recipe.get("alternative_title", ""))
                self.boost["ingredients"][r] = np.array(
                    [TITLE_BOOST if _words(names[i]) & title_words else 1.0 for i in ids]
                )

    def __len__(self):
        return len(self.recipe_ids[FEATURES[0]])

    def scores(self, feature, source_idx):
        """Similarity (0..1) of recipe `source_idx` to every recipe, for one feature."""
        ids_per_recipe = self.recipe_ids[feature]
        out = np.zeros(len(ids_per_recipe))
        source_ids = ids_per_recipe[source_idx]
        if len(source_ids) == 0:
            return out
        vecs = self.vecs[feature]
        sim = vecs[source_ids] @ vecs.T                      # source items x all distinct items
        sim = np.clip((sim - FLOOR) / (1.0 - FLOOR), 0.0, 1.0)
        idf = self.idf[feature]
        source_w = _weights(source_ids, idf, self.boost[feature][source_idx])
        for r, ids in enumerate(ids_per_recipe):
            if len(ids) == 0:
                continue
            block = sim[:, ids]                              # source items x this recipe's items
            recall = float((block.max(axis=1) * source_w).sum())     # source covered by target
            precision = float((block.max(axis=0) * _weights(ids, idf, self.boost[feature][r])).sum())  # target covered by source
            b2 = RECALL_WEIGHT ** 2
            if recall + precision > 0:
                out[r] = (1 + b2) * precision * recall / (b2 * precision + recall)
        return out

    def explain(self, feature, source_idx, target_idx):
        """For each source item: its best match in the target, similarity (0..1) and weight."""
        src, tgt = self.recipe_ids[feature][source_idx], self.recipe_ids[feature][target_idx]
        if len(src) == 0 or len(tgt) == 0:
            return []
        vecs = self.vecs[feature]
        sim = np.clip((vecs[src] @ vecs[tgt].T - FLOOR) / (1.0 - FLOOR), 0.0, 1.0)
        weights = _weights(src, self.idf[feature], self.boost[feature][source_idx])
        names = self.items[feature]
        return [(names[src[i]], names[tgt[sim[i].argmax()]], float(sim[i].max()), float(weights[i]))
                for i in range(len(src))]