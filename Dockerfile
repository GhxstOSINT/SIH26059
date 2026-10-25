FROM python:3.15.0-slim-bookworm@sha256:0e019bbdfb5d9874cb2389f1c2884b64aa2ac97c99960b5bc521cef978c0617d
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
ENV SOUTHERN_PASSAGE_VIIRS_DATA=/app/work/datasets/polarwatch-viirs \
    SOUTHERN_PASSAGE_VIIRS_SUPPLEMENT_DATA=/app/work/datasets/polarwatch-viirs-2026-sep01-21:/app/work/datasets/polarwatch-viirs-noaa21-2026-sep01-21:/app/work/datasets/polarwatch-viirs-noaa20-2026-sep01-21 \
    SOUTHERN_PASSAGE_AMSR3_DATA=/app/work/datasets/osisaf-amsr3-operational-latest \
    SOUTHERN_PASSAGE_SAR_DRIFT_DATA=/app/work/datasets/copernicus-sar-ice-drift-latest
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock
COPY service ./service
COPY ml ./ml
COPY config ./config
COPY index.html app.js review-workflow.js experience.js marine-interactions.js styles.css refinement.css product.css cinematic.css orbital.css marine.css polar-orbit.png orbital-satellite.png antarctic-sea.png hero-iceberg.png indian-research-vessel.png ./portal/
COPY portal/login.html portal/login.css ./portal/
COPY models ./models
COPY outputs ./outputs
COPY work/datasets/polarwatch-viirs ./work/datasets/polarwatch-viirs
COPY work/datasets/polarwatch-viirs-2026-sep01-21 ./work/datasets/polarwatch-viirs-2026-sep01-21
COPY work/datasets/polarwatch-viirs-noaa21-2026-sep01-21 ./work/datasets/polarwatch-viirs-noaa21-2026-sep01-21
COPY work/datasets/polarwatch-viirs-noaa20-2026-sep01-21 ./work/datasets/polarwatch-viirs-noaa20-2026-sep01-21
COPY work/datasets/osisaf-amsr3-operational-latest ./work/datasets/osisaf-amsr3-operational-latest
COPY work/datasets/copernicus-sar-ice-drift-latest ./work/datasets/copernicus-sar-ice-drift-latest
RUN mkdir -p /app/state && chown 10001:10001 /app/state
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/readyz', timeout=3).read()"
CMD ["sh", "-c", "exec uvicorn service.main:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
