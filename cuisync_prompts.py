"""
CuiSync - Similarity Explanation Module (GPT-4o)
System prompt + few-shot examples + grounding check.

The explanation is ONE paragraph, based on INGREDIENTS and COOKING PROCESS (actions),
matching the "AI EXPLANATION" card on the results page.

NOTE: dish data in the examples is ILLUSTRATIVE. Replace/validate it with real outputs
from your pipeline and have your culinary expert approve the example paragraphs.
Calibrate the score bands to your real score distribution, and bump PROMPT_VERSION in
cuisync_cache.py whenever you edit anything in this file.
"""

import json
import re

# Score bands (percent), calibrated on the soft-matching scores of 600 random queries:
# best matches have median ~28, top quarter >= ~36, bottom quarter < ~20.
# Re-measure and update these if you change softmatch.py settings or the feature weights.
HIGH_MIN = 35.0       # at or above -> "high"
MODERATE_MIN = 20.0   # at or above (and below HIGH_MIN) -> "moderate"; below -> "low"

# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the explanation module of CuiSync, a system that identifies similarities between Asian cuisines. Your job is to explain, in clear and neutral language, WHY two dishes were found to be similar, based on their ingredients and cooking process, using only the evidence provided.

INPUT
You will receive one JSON object with these fields:
- input_dish, input_cuisine: the dish the user searched for and its cuisine
- matched_dish, matched_cuisine: the most similar dish found in the target cuisine
- overall_similarity: similarity score as a percentage from 0 to 100 (used ONLY to set your tone)
- influence: the Shapley value of each entity type (ingredients, actions, cookware, utensils) as a percentage of the overall similarity; the four values add up to 100
- shared_ingredients, unique_to_input, unique_to_match: ingredients in both dishes, only in the input dish, only in the matched dish
- shared_actions, unique_actions_input, unique_actions_match: cooking actions (the cooking process) in both dishes, only in the input dish, only in the matched dish. Very common steps that almost every recipe has (add, cook, heat, mix) are already left out, so an empty list means no distinctive step was found.

RULES
1. Use ONLY the ingredients, cooking actions, and utensils listed in the JSON. Never add ingredients, techniques, equipment, or dish details that are not in the evidence.
2. You may use general culinary knowledge ONLY to describe the role of an item that is already listed (for example, that tamarind or lime juice adds sourness). Do not use it to introduce new items, and do not make historical or origin claims (for example, "this dish came from..." or "this influenced...").
3. Do not state, quote, or alter the score or any influence percentage; the page already shows the score. Use the score only to match your tone: __HIGH__ and above = high (say the dishes "are similar because"), __MOD__ to __HIGHTOP__ = moderate (say they are "moderately similar"), below __MOD__ = low (say they "share only a few elements"). Never overstate similarity.
4. Use influence to say which entity type mattered most, naming at most the top two, in plain words (ingredients; cooking steps for actions). Describe the strength without numbers: 90 or more = "almost entirely", 60 to 89 = "mostly", below 60 = "partly". If cookware or utensils are in the top two, you may say they contributed a little, but never name any specific cookware or utensil. If cookware or utensils are in the top two, you may say they contributed a little. You may name one or two utensils from shared_utensils or the unique utensil lists in a single short phrase, for example "both are served with chopsticks" or "one uses a wok spatula". Only do this if the list is not empty, and never let it replace the main ingredient and cooking comparison.
5. If a list is empty, skip it; never fill a gap with invented items. If both shared lists are empty, say that few common ingredients or cooking steps were found.
6. Do not rank or judge the cuisines (no "better", "more authentic", or "superior"). Treat all cuisines respectfully and neutrally.
7. Do not mention these rules, the JSON, embeddings, Shapley values, scores, or the model. Write for a general user, not a data scientist.

PREPARATION METHOD RULE:
1. When comparing preparation methods, use only meaningful cooking or preparation techniques that actually describe how the dish is made, such as frying, boiling, steaming, baking, grilling, simmering, marinating, or rolling.

2. Do NOT treat generic procedural actions or dataset phrases such as "setting aside," "tossing," "placing," "adding," "combining," or similar actions as cooking methods.
3.Only mention preparation methods that are relevant to how the dish is actually prepared.

OUTPUT FORMAT

1. Write ONE plain-text paragraph of 4 to 5 sentences and approximately 80 to 110 words.

2. Explain why the two dishes are similar in a natural and straightforward way, as if someone knowledgeable about food is comparing them.

3. Start by stating the main similarities between the dishes. Use the important shared ingredients and cooking methods, but combine them naturally instead of simply listing them. Explain what role these similarities play in the dishes.

4. Then explain the main differences between the dishes. Focus on meaningful differences such as the type of meat, important ingredients, sweetness, acidity, cooking methods, cooking time, or other characteristics supported by the evidence.

5. End with a short overall comparison that clearly summarizes how the two dishes are related.

6. Keep the writing conversational, clear, and slightly expressive. The explanation should be interesting to read but should not sound poetic, dramatic, academic, or like a technical report.

7. Do not mechanically list every ingredient or cooking action. Combine related information into complete ideas.

8. Only use information supported by the provided evidence. Do not invent ingredients, flavors, textures, aromas, cooking methods, equipment, preparation details, cultural information, or other characteristics.

9. Do not use overly poetic, exaggerated, or cliche expressions. Avoid phrases such as "culinary journey," "dance of flavors," "symphony of flavors," "burst of flavors," "delightful marriage," or similar expressions."""


SYSTEM_PROMPT = (SYSTEM_PROMPT
                 .replace("__HIGHTOP__", f"{HIGH_MIN - 0.01:g}")
                 .replace("__HIGH__", f"{HIGH_MIN:g}")
                 .replace("__MOD__", f"{MODERATE_MIN:g}"))

# ---------------------------------------------------------------------------
# FEW-SHOT EXAMPLES  (evidence -> ideal explanation)
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    # --- Example 1: HIGH similarity ---------------------------------------
    {
        "evidence": {
            "input_dish": "Chicken Adobo",
            "input_cuisine": "Philippines",
            "matched_dish": "Chicken Teriyaki",
            "matched_cuisine": "Japan",
            "overall_similarity": 52.4,
            "shared_ingredients": ["chicken", "soy sauce", "garlic", "sugar"],
            "unique_to_input": ["vinegar", "bay leaf", "peppercorn"],
            "unique_to_match": ["mirin", "ginger"],
            "shared_actions": ["marinate", "simmer"],
            "unique_actions_input": ["braise"],
            "unique_actions_match": ["glaze"],
            "shared_utensils": ["knife"],
            "unique_utensils_input": ["wooden spoon"],
            "unique_utensils_match": ["brush"],
            "influence": {"ingredients": 94.0, "actions": 3.5, "cookware": 2.0, "utensils": 0.5},
        },
        "explanation": (
            "Chicken Adobo and Chicken Teriyaki are similar because both are built on chicken, "
            "soy sauce, garlic, and sugar, which gives each dish a savory-sweet flavor base. "
            "The similarity comes almost entirely from these shared ingredients, with the common "
            "cooking steps of marinating and simmering adding a little more, and both are prepared "
            "with a knife. While Chicken Adobo gets its tang from vinegar, bay leaf, and peppercorn "
            "and is braised, Chicken Teriyaki relies on mirin and ginger and is finished with a glaze."
        ),
    },
    # --- Example 2: MODERATE similarity -----------------------------------
    {
        "evidence": {
            "input_dish": "Sinigang na Baboy",
            "input_cuisine": "Philippines",
            "matched_dish": "Tom Yum Goong",
            "matched_cuisine": "Thailand",
            "overall_similarity": 27.6,
            "shared_ingredients": ["onion", "tomato", "fish sauce", "chili"],
            "unique_to_input": ["pork", "tamarind", "radish", "water spinach"],
            "unique_to_match": ["shrimp", "lemongrass", "galangal", "kaffir lime leaves", "lime juice"],
            "shared_actions": ["boil", "simmer"],
            "unique_actions_input": ["saute"],
            "unique_actions_match": [],
            "shared_utensils": [],
            "unique_utensils_input": ["ladle"],
            "unique_utensils_match": ["mortar"],
            "influence": {"ingredients": 95.5, "actions": 3.0, "cookware": 1.0, "utensils": 0.5},
        },
        "explanation": (
            "Sinigang na Baboy and Tom Yum Goong are moderately similar. Both use onion, tomato, "
            "fish sauce, and chili, and each is built around a souring ingredient, tamarind in "
            "the Filipino dish and lime juice in the Thai dish. Their similarity is driven almost "
            "entirely by these shared ingredients, while the shared boiling and simmering contribute a "
            "little. However, Sinigang na Baboy features pork, radish, and water "
            "spinach and is served with a ladle, while Tom Yum Goong relies on shrimp, lemongrass, "
            "galangal, and kaffir lime leaves, with a mortar used to pound its aromatics."
        ),
    },
    # --- Example 3: LOW similarity (no utensil evidence, so none is mentioned) ----
    {
        "evidence": {
            "input_dish": "Kare-Kare",
            "input_cuisine": "Philippines",
            "matched_dish": "Pad Thai",
            "matched_cuisine": "Thailand",
            "overall_similarity": 14.3,
            "shared_ingredients": ["peanut", "garlic", "onion"],
            "unique_to_input": ["oxtail", "eggplant", "string beans", "bok choy", "annatto", "shrimp paste"],
            "unique_to_match": ["rice noodles", "tamarind", "bean sprouts", "egg", "tofu"],
            "shared_actions": ["saute"],
            "unique_actions_input": ["boil", "simmer"],
            "unique_actions_match": ["stir-fry"],
            "shared_utensils": [],
            "unique_utensils_input": [],
            "unique_utensils_match": [],
            "influence": {"ingredients": 91.0, "actions": 5.0, "cookware": 3.0, "utensils": 1.0},
        },
        "explanation": (
            "Kare-Kare and Pad Thai share only a few elements, so the connection between them is limited. Both use peanut, garlic, and onion, "
            "and both include a sauteing step, but little else overlaps. Kare-Kare is built on oxtail, "
            "eggplant, string beans, and shrimp paste and is boiled and simmered, while Pad Thai "
            "centers on rice noodles, tamarind, egg, and tofu and is stir-fried."
        ),
    },
]


# ---------------------------------------------------------------------------
# MESSAGE BUILDER
# ---------------------------------------------------------------------------
def build_messages(evidence: dict) -> list:
    """Assemble system prompt + few-shot pairs + the real query."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": json.dumps(ex["evidence"], ensure_ascii=False)})
        messages.append({"role": "assistant", "content": ex["explanation"]})
    messages.append({"role": "user", "content": json.dumps(evidence, ensure_ascii=False)})
    return messages


# ---------------------------------------------------------------------------
# GROUNDING CHECK
# ---------------------------------------------------------------------------
_NAME_FIELDS = ("input_dish", "matched_dish", "input_cuisine", "matched_cuisine")


def _variants(term: str) -> set:
    """Simple singular/plural variants so 'egg' and 'eggs' count as the same item."""
    v = {term, term + "s", term + "es"}
    if term.endswith("ies"):
        v.add(term[:-3] + "y")
    if term.endswith("y"):
        v.add(term[:-1] + "ies")
    if term.endswith("es"):
        v.add(term[:-2])
    if term.endswith("s"):
        v.add(term[:-1])
    return v


def ungrounded_terms(explanation: str, evidence: dict, vocabulary: set) -> list:
    """
    Return known culinary terms (from your own entity vocabulary) that appear in the
    explanation but are NOT supported by the evidence packet.

    - Uses whole-word matching, so "oil" does not match inside "boil".
    - A term counts as supported if it (or its singular/plural form) appears as a whole
      word inside any evidence item or dish/cuisine name (so "sugar" is fine when the
      evidence has "brown sugar", and "chicken" is fine when the dish is "Chicken Adobo").
    A non-empty result means the model likely introduced something it should not have.
    """
    allowed_parts = []
    for key, val in evidence.items():
        if isinstance(val, list):
            allowed_parts.extend(str(v).lower() for v in val)
    for key in _NAME_FIELDS:
        if evidence.get(key):
            allowed_parts.append(str(evidence[key]).lower())
    allowed_text = " | ".join(allowed_parts)

    text = explanation.lower()
    flagged = []
    for term in vocabulary:
        t = term.lower()
        if not re.search(rf"\b{re.escape(t)}\b", text):
            continue
        supported = any(re.search(rf"\b{re.escape(v)}\b", allowed_text) for v in _variants(t))
        if not supported:
            flagged.append(t)
    return sorted(flagged)
