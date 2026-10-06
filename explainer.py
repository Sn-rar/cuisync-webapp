"""
CuiSync - Similarity explanation module (GPT-4o)

Everything about generating the "AI EXPLANATION" paragraph lives here,
so app.py only handles routes and similarity computation.

Flow:  recipes -> cleaned evidence -> cache -> GPT-4o -> grounding check -> text
       (any failure -> template fallback; failed answers are never cached)

The evidence is CLEANED before it reaches GPT (the chips on the results page are not
affected). The cleaning fixes problems in the extracted entities that make explanations
weak:
  - typos            "fisf sauce" -> "fish sauce", "bayleaf" -> "bay leaf"
  - generic steps    "add", "cook", "heat", "mix" (in 30%+ of all recipes) say nothing
  - false differences  "chicken drumsticks" vs "chicken meat" are both just chicken;
                       "oil" vs "olive oil", "chili peppers" vs "red chili pepper"

Needs the OPENAI_API_KEY environment variable (or a .env file next to app.py).
Without it, the app still works and shows a template explanation instead.
The terminal prints at startup whether GPT is ENABLED or DISABLED, and why.
"""

import difflib
import logging
import os
import re
from collections import Counter
from functools import lru_cache

try:                                   # optional: read OPENAI_API_KEY from a .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from cuisync_cache import explain_with_cache, save_cached, MODEL_NAME
from cuisync_prompts import build_messages, ungrounded_terms, HIGH_MIN, MODERATE_MIN

try:
    from openai import OpenAI
except ImportError:                    # package not installed -> template fallback only
    OpenAI = None

log = logging.getLogger(__name__)

_API_KEY = os.environ.get("OPENAI_API_KEY")
_client = OpenAI(api_key=_API_KEY, timeout=20.0, max_retries=1) if (OpenAI and _API_KEY) else None

if _client is not None:
    print("[CuiSync] GPT-4o explanations: ENABLED")
elif OpenAI is None:
    print("[CuiSync] GPT-4o explanations: DISABLED - the 'openai' package is not installed "
          "(run: pip install -r requirements.txt). Using template explanations.")
else:
    print("[CuiSync] GPT-4o explanations: DISABLED - OPENAI_API_KEY was not found "
          "(set it in this terminal or in a .env file; see README step 5). "
          "Using template explanations.")


# ---------------------------------------------------------------------------
# Dataset statistics (built once at startup by init())
# ---------------------------------------------------------------------------
GENERIC_ACTION_SHARE = 0.30     # actions in >= 30% of recipes are treated as generic
MAX_SHARED = 8
MAX_UNIQUE = 6
MAX_ACTIONS = 4

# Basic items that carry no information about a dish's identity.
GENERIC_INGREDIENTS = {"water", "salt", "ice"}

# Generic words that appear as "entities" in the dataset but are also ordinary prose.
# They are excluded from the grounding vocabulary so valid explanations are not rejected.
GENERIC_WORDS = {
    "sauce", "sweet", "food", "deep", "red", "green", "white", "black", "hot", "cold",
    "fresh", "meat", "dish", "small", "large", "the", "and", "are", "use", "mix",
    "heat", "cooking", "flavor", "well", "then", "more", "base", "ice", "rock",
    "bird", "acid", "bath", "coal", "water",
    # cuisine adjectives GPT uses naturally ("the Thai dish")
    "thai", "chinese", "japanese", "korean", "indian", "filipino", "vietnamese",
    "indonesian", "malaysian", "burmese", "cambodian", "khmer", "turkish", "saudi",
    "arab", "arabic", "asian", "western",
    # cooking verbs that also appear as noisy "items" in the dataset
    "add", "baking", "brush", "grate", "ground", "julienne", "mixing", "roasting",
    "saute", "scoop", "seasoning", "steam", "steep", "whisk", "cup",
}

# Words that describe an item without changing what it is.
_MODIFIERS = {
    "fresh", "frozen", "cooked", "raw", "ground", "chopped", "sliced", "diced", "minced",
    "dried", "red", "green", "white", "black", "yellow", "small", "large", "whole", "fine",
    "finely", "hot", "sweet", "ripe", "extra", "light", "dark", "boneless", "skinless",
}
_SUBSET_EXTRAS = _MODIFIERS | {"olive", "vegetable", "canola", "cooking"}

# Main proteins: "chicken drumsticks" and "chicken meat" are both just chicken.
_PROTEINS = {"chicken", "pork", "beef", "shrimp", "prawn", "fish", "egg", "tofu", "lamb",
             "mutton", "duck", "squid", "crab", "goat", "turkey"}
# ...unless the item is really a product made from it ("fish sauce", "egg noodle").
_NOT_PROTEIN = {"sauce", "paste", "oil", "stock", "broth", "powder", "cube", "flake", "ball",
                "cake", "noodle", "wrapper", "roll", "vinegar", "juice", "mayonnaise"}

# Real cooking methods are more telling than prep verbs ("cut", "rinse"), so they rank first.
_METHODS = {"boil", "simmer", "fry", "stir-fry", "deep-fry", "pan-fry", "saute", "sauté", "grill",
            "roast", "bake", "steam", "braise", "stew", "marinate", "ferment", "smoke", "poach",
            "blanch", "toast", "sear", "broil", "barbecue", "pickle", "caramelize", "pressure cook"}

_PARSE = None
_ITEM_COUNTS = Counter()
_ACTION_SHARE = {}
_GENERIC_ACTIONS = set()
_FREQUENT_ITEMS = []
_VOCAB = set()


def build_vocabulary(recipes, fields, parse_fn):
    """
    Terms from the given recipe fields (lowercase), used by the grounding check to spot
    physical items (ingredients, cookware, utensils) that GPT mentions but that are not
    in the evidence. Do NOT pass the action field: it contains noisy words ("and", "use").
    """
    vocab = set()
    for recipe in recipes:
        for field in fields:
            for item in parse_fn(recipe.get(field) or ""):
                t = item.lower()
                if len(t) >= 3 and t not in GENERIC_WORDS:
                    vocab.add(t)
    return vocab


def init(recipes, parse_fn):
    """Call once at startup: learns item frequencies, generic steps, and the vocabulary."""
    global _PARSE, _ITEM_COUNTS, _ACTION_SHARE, _GENERIC_ACTIONS, _FREQUENT_ITEMS, _VOCAB
    _PARSE = parse_fn
    items, actions = Counter(), Counter()
    for r in recipes:
        items.update({i.lower() for i in parse_fn(r.get("ingredient_text") or "")})
        actions.update({a.lower() for a in parse_fn(r.get("action_text") or "")})
    n = max(len(recipes), 1)
    _ITEM_COUNTS = items
    _ACTION_SHARE = {a: c / n for a, c in actions.items()}
    _GENERIC_ACTIONS = {a for a, share in _ACTION_SHARE.items() if share >= GENERIC_ACTION_SHARE}
    _FREQUENT_ITEMS = [i for i, c in items.items() if c >= 5]
    _VOCAB = build_vocabulary(recipes, ["ingredient_text", "cookware_text", "utensil_text"], parse_fn)
    _fix_typo.cache_clear()


# ---------------------------------------------------------------------------
# Evidence cleaning
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4096)
def _fix_typo(item):
    """Map a rare item to a frequent, very similar one ('fisf sauce' -> 'fish sauce')."""
    if len(item) < 5 or _ITEM_COUNTS.get(item, 0) > 2:
        return item
    match = difflib.get_close_matches(item, _FREQUENT_ITEMS, n=1, cutoff=0.88)
    return match[0] if match else item


def _singular(word):
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens(item):
    return [_singular(t) for t in re.findall(r"[a-z]+", item.lower())]


def _related(a, b):
    """
    If two different items are really the same thing, return the common concept, else None.
      'oil' / 'olive oil'                   -> 'oil'
      'chili peppers' / 'red chili pepper'  -> 'chili peppers'
      'chicken drumsticks' / 'chicken meat' -> 'chicken'
    """
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return None
    small, big, short_item = (ta, tb, a) if len(ta) <= len(tb) else (tb, ta, b)
    if small <= big and (big - small) <= _SUBSET_EXTRAS:
        return short_item
    if not ((ta | tb) & _NOT_PROTEIN):
        common = ta & tb & _PROTEINS
        if common:
            return sorted(common)[0]
    return None


def _compare_ingredients(src_items, tgt_items):
    """Returns (shared, only_in_source, only_in_target) after cleaning."""
    def prep(items):
        return sorted({_fix_typo(i.lower()) for i in items} - GENERIC_INGREDIENTS)

    src, tgt = prep(src_items), prep(tgt_items)
    shared = set(src) & set(tgt)
    concepts = set()

    def split(mine, other):
        keep = []
        for item in mine:
            if item in shared:
                continue
            concept = next((c for c in (_related(item, o) for o in other) if c), None)
            if concept:
                concepts.add(concept)
            else:
                keep.append(item)
        return keep

    only_src, only_tgt = split(src, tgt), split(tgt, src)

    for concept in concepts:                        # add the common concept (e.g. "chicken")
        ctoks = set(_tokens(concept))               # unless a shared item already covers it
        if not any(ctoks <= set(_tokens(s)) for s in shared):
            shared.add(concept)

    rare_first = lambda x: (_ITEM_COUNTS.get(x, 0), x)      # distinctive items first
    common_first = lambda x: (-_ITEM_COUNTS.get(x, 0), x)   # recognisable items first
    return (sorted(shared, key=rare_first)[:MAX_SHARED],
            sorted(only_src, key=common_first)[:MAX_UNIQUE],
            sorted(only_tgt, key=common_first)[:MAX_UNIQUE])


def _compare_actions(src_items, tgt_items):
    """Same idea for cooking actions; generic steps (add, cook, heat, ...) are dropped."""
    def prep(items):
        return {a.lower().strip() for a in items} - _GENERIC_ACTIONS

    src, tgt = prep(src_items), prep(tgt_items)
    order = lambda x: (0 if x in _METHODS else 1, -_ACTION_SHARE.get(x, 0), x)
    return (sorted(src & tgt, key=order)[:MAX_ACTIONS],
            sorted(src - tgt, key=order)[:MAX_ACTIONS],
            sorted(tgt - src, key=order)[:MAX_ACTIONS])


def build_evidence(source_dish, recommended_dish, similarity_pct, influence):
    """
    Evidence packet sent to GPT-4o: the overall score, the Shapley influence of each entity
    type, and the CLEANED ingredient and cooking-action comparison.
    """
    parse = _PARSE or (lambda t: [x.strip() for x in (t or "").split(",") if x.strip()])
    s_ing, only_s, only_t = _compare_ingredients(
        parse(source_dish.get("ingredient_text", "")), parse(recommended_dish.get("ingredient_text", "")))
    s_act, act_s, act_t = _compare_actions(
        parse(source_dish.get("action_text", "")), parse(recommended_dish.get("action_text", "")))
    ut_s = {u.lower() for u in parse(source_dish.get("utensil_text", ""))}
    ut_t = {u.lower() for u in parse(recommended_dish.get("utensil_text", ""))}

    evidence = {
        "input_dish": source_dish["title"].strip(),
        "input_cuisine": source_dish["country"].title(),
        "matched_dish": recommended_dish["title"].strip(),
        "matched_cuisine": recommended_dish["country"].title(),
        "overall_similarity": similarity_pct,
        "shared_ingredients": s_ing,
        "unique_to_input": only_s,
        "unique_to_match": only_t,
        "shared_actions": s_act,
        "unique_actions_input": act_s,
        "unique_actions_match": act_t,
        "shared_utensils": sorted(ut_s & ut_t)[:4],
        "unique_utensils_input": sorted(ut_s - ut_t)[:3],
        "unique_utensils_match": sorted(ut_t - ut_s)[:3],
    }
    if influence:
        evidence["influence"] = influence
    return evidence


# ---------------------------------------------------------------------------
# GPT call, fallback, orchestration
# ---------------------------------------------------------------------------
def _call_gpt(evidence, avoid=None):
    """Raises on API errors, so failures are never cached.
    avoid: items the previous draft mentioned that are not in the evidence (retry only)."""
    try:
        messages = build_messages(evidence)
        if avoid:
            messages.append({"role": "user", "content": (
                "Your previous draft mentioned items that are not in the evidence: "
                + ", ".join(avoid)
                + ". Write the paragraph again using ONLY the ingredients, cooking actions and utensils"
                  "in the JSON above, and do not mention those items.")})
        resp = _client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,
            max_tokens=250,
        )

        print("[GPT RESPONSE]", resp.choices[0].message.content)

        return resp.choices[0].message.content.strip()

    except Exception as exc:
        print("[GPT ERROR]", repr(exc))
        raise


def fallback_explanation(ev):
    """Template paragraph used when the API is unavailable or fails the grounding check."""
    score = ev["overall_similarity"]
    a, b = ev["input_dish"], ev["matched_dish"]

    def join(items):
        items = list(items)
        if len(items) <= 1:
            return "".join(items)
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + ", and " + items[-1]

    if ev["shared_ingredients"]:
        shared = join(ev["shared_ingredients"][:5])
        if score >= HIGH_MIN:
            opener = f"{a} and {b} are similar because both use {shared}."
        elif score >= MODERATE_MIN:
            opener = f"{a} and {b} are moderately similar. Both use {shared}."
        else:
            opener = f"{a} and {b} share only a few elements. Both use {shared}."
    else:
        opener = f"{a} and {b} share few common ingredients."

    parts = [opener]
    if ev["shared_actions"]:
        parts.append(f"They also share the cooking steps of {join(ev['shared_actions'])}.")

    diffs = []
    if ev["unique_to_input"]:
        diffs.append(f"{a} also uses {join(ev['unique_to_input'][:4])}")
    if ev["unique_to_match"]:
        diffs.append(f"{b} also uses {join(ev['unique_to_match'][:4])}")
    if diffs:
        parts.append("However, " + ", while ".join(diffs) + ".")

    return " ".join(parts)


def get_explanation(evidence):
    if _client is None:
        print("[CuiSync] GPT unavailable -> TEMPLATE")
        return fallback_explanation(evidence), "template"

    def grounded(text, ev):
        bad = ungrounded_terms(text, ev, _VOCAB)

        if bad:
            print("[CuiSync] GROUNDING FAILED")
            print("[CuiSync] Unsupported terms:", bad)
            return False

        print("[CuiSync] GROUNDING PASSED")
        return True

    try:
        text = explain_with_cache(evidence, _call_gpt, grounded)

        bad = ungrounded_terms(text, evidence, _VOCAB)

        if not bad:
            print("[CuiSync] GPT ACCEPTED")
            return text, "ai"

        print("[CuiSync] GPT REJECTED -> retrying once without:", bad)

        # One retry: tell GPT which items it must leave out. A passing answer is cached.
        retry = _call_gpt(evidence, avoid=bad)
        if not ungrounded_terms(retry, evidence, _VOCAB):
            save_cached(evidence, retry)
            print("[CuiSync] GPT ACCEPTED (retry)")
            return retry, "ai"
        print("[CuiSync] GPT REJECTED AGAIN:", ungrounded_terms(retry, evidence, _VOCAB))

    except Exception as exc:
        print("[CuiSync] GPT API FAILED:", type(exc).__name__, str(exc))

    print("[CuiSync] TEMPLATE FALLBACK")
    return fallback_explanation(evidence), "template"