FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock
COPY service ./service
COPY ml ./ml
COPY config ./config
COPY index.html app.js review-workflow.js experience.js marine-interactions.js styles.css refinement.css product.css cinematic.css orbital.css marine.css polar-orbit.png orbital-satellite.png antarctic-sea.png hero-iceberg.png indian-research-vessel.png ./portal/
COPY models ./models
COPY outputs ./outputs
RUN mkdir -p /app/state && chown 10001:10001 /app/state
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/readyz', timeout=3).read()"
CMD ["sh", "-c", "exec uvicorn service.main:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
