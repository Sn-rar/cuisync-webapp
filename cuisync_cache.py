"""
CuiSync - Explanation cache (SQLite)

Stores GPT-4o explanations so the same dish pair is only sent to the API once.
Uses only Python's standard library (sqlite3, hashlib, json).
"""

import hashlib
import json
import os
import sqlite3

# Stored next to this file, so it works no matter where the app is launched from.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "explanations_cache.db")

# Bump these when you change the prompt or model, so old cached answers
# are not reused with the new setup.
PROMPT_VERSION = "v8"
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o")


def _init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS explanations (
                cache_key   TEXT PRIMARY KEY,
                input_dish  TEXT,
                matched_dish TEXT,
                explanation TEXT NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def make_cache_key(evidence: dict) -> str:
    """
    Build a stable key from the evidence packet.
    Lists are sorted so the same items in a different order give the same key.
    """
    normalized = {}
    for k, v in evidence.items():
        if isinstance(v, list):
            normalized[k] = sorted(v)
        else:
            normalized[k] = v
    raw = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    raw += f"|{PROMPT_VERSION}|{MODEL_NAME}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached(evidence: dict):
    """Return the cached explanation text, or None if not cached."""
    key = make_cache_key(evidence)
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT explanation FROM explanations WHERE cache_key = ?", (key,)
        ).fetchone()
    return row[0] if row else None


def save_cached(evidence: dict, explanation: str):
    key = make_cache_key(evidence)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO explanations "
            "(cache_key, input_dish, matched_dish, explanation) VALUES (?, ?, ?, ?)",
            (key, evidence.get("input_dish"), evidence.get("matched_dish"), explanation),
        )


def explain_with_cache(evidence: dict, generate_fn, is_valid_fn=None) -> str:
    """
    1. Return the cached explanation if one exists.
    2. Otherwise call generate_fn(evidence) (your GPT-4o call).
    3. Save it only if it passes is_valid_fn (e.g., your grounding check).
    """
    cached = get_cached(evidence)
    if cached is not None:
        return cached

    explanation = generate_fn(evidence)

    if is_valid_fn is None or is_valid_fn(explanation, evidence):
        save_cached(evidence, explanation)
    return explanation


_init_db()


# ---------------------------------------------------------------------------
# USAGE (Flask route)
# ---------------------------------------------------------------------------
# from cuisync_prompts import build_messages, ungrounded_terms
#
# def call_gpt(evidence):
#     resp = client.chat.completions.create(
#         model=MODEL_NAME, messages=build_messages(evidence),
#         temperature=0.2, max_tokens=350)
#     return resp.choices[0].message.content
#
# def passes_grounding(explanation, evidence):
#     return not ungrounded_terms(explanation, evidence, ENTITY_VOCABULARY)
#
# @app.route("/explain", methods=["POST"])
# def explain():
#     evidence = build_evidence(request.json)          # from your similarity module
#     text = explain_with_cache(evidence, call_gpt, passes_grounding)
#     return jsonify({"explanation": text})
