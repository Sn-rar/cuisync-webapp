FROM python:3.11-slim

# Libraries OpenCV and Paddle need on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face runs the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Download all the OCR models now, so waking the Space up doesn't have to
RUN python -c "import ocr; ocr.warm_up()"

EXPOSE 7860
CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "300"]
