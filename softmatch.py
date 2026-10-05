"""Soft ingredient matching for Cuisync (put this file next to app.py).

For two recipes A and B, every ingredient in A is matched to its CLOSEST ingredient in B
(by language-model similarity) and vice-versa, so 'chicken breast' ~ 'chicken thigh' and
'ginger' ~ 'ginger root' count as near-matches, while 'chicken' vs 'pork' counts for little.

Changes vs the previous version (API unchanged: SoftMatcher(folder, recipes), scores(),
explain(), len()):
  * scores() is vectorised across all recipes (same numbers, much faster, weights are
    computed once at start-up instead of once per recipe per query).
  * scores(..., beta=None) lets you override RECALL_WEIGHT per call.
  * pair_score(): a SYMMETRIC A<->B score (beta=1 by default) for display. With
    RECALL_WEIGHT=2 the ranking score is asymmetric, so score(A->B) != score(B->A); showing
    that number as "the similarity between A and B" is inconsistent.
  * similarity_matrix() / csls(): optional full N x N matrix and hubness correction.
"""
import json
import os
import re

import numpy as np

FEATURES = ("ingredients", "actions", "cookware", "utensils")

# Similarities at or below FLOOR count as 0 (unrelated); the rest are stretched to 0..1.
# If you use CENTER_VECTORS=True in the vectorizer, unrelated pairs sit near 0 already, so a
# much lower FLOOR (about 0.05-0.2) is right. Tune it with evaluate_softmatch.py.
FLOOR = 0.35

# 0 = every item counts equally. If your lists are written main-ingredient-first, try 0.15-0.3:
# item at position k gets weight 1 / (1 + POSITION_DECAY * k), so early items matter more.
POSITION_DECAY = 0.0

# Items that appear in many recipes (salt, oil, water, garlic...) say little about a dish; rare,
# defining items say a lot. IDF_POWER scales that effect: 0 = ignore, 1 = standard, 2 = stronger.
IDF_POWER = 1.0

# How much more "is the source dish's makeup covered by the target?" (recall) counts than
# "is everything in the target found in the source?" (precision). 1 = equal (plain F1).
RECALL_WEIGHT = 2.0

# An ingredient whose word also appears in the dish TITLE ("chicken" in "Fried Chicken") is the
# dish's headline ingredient, so it counts this many times more. 1 = off. Ingredients only.
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


def _fbeta(precision, recall, beta):
    """F-beta; beta > 1 favours recall. Works on scalars or arrays; 0 where undefined."""
    b2 = beta ** 2
    num = (1 + b2) * precision * recall
    den = b2 * precision + recall
    return np.divide(num, den, out=np.zeros_like(np.asarray(den, dtype=np.float64)), where=den > 0)


class SoftMatcher:
    def __init__(self, folder, recipes=None):
        self.vecs, self.recipe_ids, self.idf, self.items, self.boost = {}, {}, {}, {}, {}
        for feature in FEATURES:
            self.vecs[feature] = np.load(os.path.join(folder, f"{feature}_item_vecs.npy"))
            with open(os.path.join(folder, f"{feature}_items.json"), encoding="utf-8") as f:
                data = json.load(f)
            self.recipe_ids[feature] = [np.array(ids, dtype=np.int64) for ids in data["recipes"]]
            if "df" in data:                                   # written by the vectorizer
                df = np.array(data["df"], dtype=np.float64)
            else:                                              # older export: count it here
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

        # Per-recipe weights, computed once. Also a flattened layout (all non-empty recipes'
        # items back to back) so a query can be scored for every recipe in a few numpy calls.
        self.weights, self._flat = {}, {}
        for feature in FEATURES:
            ids_list = self.recipe_ids[feature]
            self.weights[feature] = [
                _weights(ids, self.idf[feature], self.boost[feature][r]) if len(ids) else np.zeros(0)
                for r, ids in enumerate(ids_list)
            ]
            nonempty = np.flatnonzero([len(ids) > 0 for ids in ids_list])
            if len(nonempty):
                flat_ids = np.concatenate([ids_list[r] for r in nonempty])
                flat_w = np.concatenate([self.weights[feature][r] for r in nonempty])
                lens = np.array([len(ids_list[r]) for r in nonempty])
                starts = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
            else:
                flat_ids, flat_w, starts = np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64)
            self._flat[feature] = (flat_ids, flat_w, starts, nonempty)

    def __len__(self):
        return len(self.recipe_ids[FEATURES[0]])

    def _sim(self, a_ids, b_ids, feature):
        v = self.vecs[feature]
        return np.clip((v[a_ids] @ v[b_ids].T - FLOOR) / (1.0 - FLOOR), 0.0, 1.0)

    def scores(self, feature, source_idx, beta=None):
        """Similarity (0..1) of recipe `source_idx` to every recipe, for one feature.
        Asymmetric when beta != 1 (source coverage counts beta^2 times more than target)."""
        beta = RECALL_WEIGHT if beta is None else beta
        out = np.zeros(len(self.recipe_ids[feature]))
        src = self.recipe_ids[feature][source_idx]
        flat_ids, flat_w, starts, nonempty = self._flat[feature]
        if len(src) == 0 or len(flat_ids) == 0:
            return out
        vecs = self.vecs[feature]
        sim_v = np.clip((vecs[src] @ vecs.T - FLOOR) / (1.0 - FLOOR), 0.0, 1.0)   # source x distinct items
        sim = sim_v[:, flat_ids]                                                  # source x every recipe item
        recall = self.weights[feature][source_idx] @ np.maximum.reduceat(sim, starts, axis=1)
        precision = np.add.reduceat(sim.max(axis=0) * flat_w, starts)
        out[nonempty] = _fbeta(precision, recall, beta)
        return out

    def pair_score(self, feature, a_idx, b_idx, beta=1.0):
        """Score for one pair of recipes. beta=1 (default) is symmetric: A,B == B,A.
        Use this for the number you display as 'similarity between A and B'."""
        a, b = self.recipe_ids[feature][a_idx], self.recipe_ids[feature][b_idx]
        if len(a) == 0 or len(b) == 0:
            return 0.0
        sim = self._sim(a, b, feature)
        recall = float((sim.max(axis=1) * self.weights[feature][a_idx]).sum())
        precision = float((sim.max(axis=0) * self.weights[feature][b_idx]).sum())
        return float(_fbeta(precision, recall, beta))

    def explain(self, feature, source_idx, target_idx):
        """For each source item: its best match in the target, similarity (0..1) and weight."""
        src, tgt = self.recipe_ids[feature][source_idx], self.recipe_ids[feature][target_idx]
        if len(src) == 0 or len(tgt) == 0:
            return []
        sim = self._sim(src, tgt, feature)
        weights = self.weights[feature][source_idx]
        names = self.items[feature]
        return [(names[src[i]], names[tgt[sim[i].argmax()]], float(sim[i].max()), float(weights[i]))
                for i in range(len(src))]

    def similarity_matrix(self, feature, beta=1.0):
        """Full N x N matrix (row = source). Precompute once if you want hubness correction
        or percentile-calibrated display scores."""
        n = len(self)
        return np.stack([self.scores(feature, i, beta=beta) for i in range(n)])


def csls(matrix, k=10):
    """Hubness correction (CSLS). Dishes that are 'close to everything' (generic stews, plain
    fried rice) get their score reduced; distinctive dishes are not punished. The result is a
    ranking score, not 0..1 - convert to percentiles per row before showing it to users."""
    m = np.array(matrix, dtype=np.float64)
    masked = m.copy()
    np.fill_diagonal(masked, -np.inf)
    k = min(k, m.shape[1] - 1)
    r_row = np.sort(np.partition(masked, -k, axis=1)[:, -k:], axis=1).mean(axis=1)
    r_col = np.sort(np.partition(masked, -k, axis=0)[-k:, :], axis=0).mean(axis=0)
    return 2.0 * m - r_row[:, None] - r_col[None, :]