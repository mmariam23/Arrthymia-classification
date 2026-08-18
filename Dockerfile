FROM python:3.11-slim

# HF Spaces runs as non-root user uid 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY main.py      .
COPY app.py       .
COPY pipeline/    ./pipeline/

# Model dir (mount your checkpoint here at runtime)
RUN mkdir -p model && chown -R appuser:appuser /app

USER appuser

ENV MODEL_PATH=model/vgg_arrhythmia.pt

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]