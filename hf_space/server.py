"""CuiSync OCR service (runs on the Hugging Face Space).

It only reads menu photos. The website itself runs on Render and sends
photos here.

    GET  /health  -> {"status": "ok", "models_ready": true/false}
    POST /ocr     -> form fields: image (file), country (optional)
                     answer: {"lines": [...]} or
                             {"lines": [...], "error": "unsupported_script", "message": "..."}

If the OCR_API_TOKEN secret is set on the Space, requests must send
the header "Authorization: Bearer <that token>".
"""
import hmac
import os
import threading

# The Space always runs PaddleOCR itself, never forwards to another service.
os.environ.pop("OCR_API_URL", None)

import cv2
import numpy as np
from flask import Flask, jsonify, request

import ocr

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

TOKEN = os.environ.get("OCR_API_TOKEN", "").strip()
MODELS_READY = threading.Event()


def _load_models():
    ocr.warm_up()
    MODELS_READY.set()


threading.Thread(target=_load_models, daemon=True).start()


def _authorized():
    if not TOKEN:
        return True
    sent = request.headers.get("Authorization", "")
    if sent.startswith("Bearer "):
        sent = sent[len("Bearer "):]
    return hmac.compare_digest(sent.strip(), TOKEN)


@app.get("/")
@app.get("/health")
def health():
    return jsonify({"status": "ok", "models_ready": MODELS_READY.is_set()})


@app.post("/ocr")
def read_menu():
    if not _authorized():
        return jsonify({"error": "unauthorized", "message": "Wrong or missing OCR token."}), 401

    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "no_image", "message": "No image uploaded."}), 400

    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "bad_image", "message": "Image format is not supported or corrupted."}), 400

    country = (request.form.get("country") or "").strip()
    try:
        if country:
            lines = ocr.automatic_ocr_for_country(image, country)
        else:
            lines = ocr.automatic_ocr(image)
    except ocr.UnsupportedScriptError as error:
        return jsonify({"lines": error.lines, "error": "unsupported_script", "message": str(error)})
    except Exception as error:
        app.logger.exception("OCR failed")
        return jsonify({"error": "ocr_failed", "message": f"OCR failed: {error}"}), 500

    return jsonify({"lines": lines})
