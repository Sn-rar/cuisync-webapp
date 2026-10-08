---
title: CuiSync OCR
sdk: docker
app_port: 7860
pinned: false
---

# CuiSync OCR service

Reads menu photos with PaddleOCR (PP-OCRv5) for the CuiSync web app.

Files in this Space: `Dockerfile`, `server.py`, `requirements.txt`, this `README.md`,
and a copy of `ocr.py` from the main CuiSync project.

Set a secret named `OCR_API_TOKEN` in the Space settings; the web app must send the same value.
