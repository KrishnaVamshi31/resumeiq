# syntax=docker/dockerfile:1
#
# Multi-stage build: dependencies are resolved once in the builder and the
# runtime image carries only the virtualenv and the application. The result
# runs as a non-root user with no build toolchain present.

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Copy only what pip needs to resolve dependencies, so the layer caches until
# the dependency set actually changes.
COPY pyproject.toml README.md ./
COPY app/__init__.py app/__init__.py

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel \
    && pip install "fastapi>=0.115.0" "uvicorn[standard]>=0.32.0" "pydantic>=2.9.0" \
       "pydantic-settings>=2.6.0" "python-multipart>=0.0.12" "SQLAlchemy>=2.0.36" \
       "pypdf>=5.1.0" "python-docx>=1.1.2" "anthropic>=1.0.0"


FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RESUMEIQ_ENV=production \
    RESUMEIQ_LOG_FORMAT=json \
    RESUMEIQ_DATABASE_URL=sqlite:////data/resumeiq.db

# curl is used by the container healthcheck below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 resumeiq

COPY --from=builder /opt/venv /opt/venv

WORKDIR /srv/resumeiq
COPY app ./app
COPY data ./data
COPY pyproject.toml README.md ./

# The database lives on a volume so it survives container replacement.
RUN mkdir -p /data && chown -R resumeiq:resumeiq /data /srv/resumeiq

USER resumeiq

# Managed platforms inject the port to bind as $PORT and route only to it;
# `Settings.port` reads it, and this default covers plain `docker run`.
ENV PORT=8000
EXPOSE 8000
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# One worker per container; scale with replicas so the in-process rate limiter
# and metrics registry stay coherent per instance.
#
# Booting through the project CLI rather than the uvicorn binary: `serve`
# passes log_config=None, which leaves our JSON logging in place. The uvicorn
# CLI has no equivalent -- `--log-config` always hands the path to
# logging.config.fileConfig, and an empty file (/dev/null) makes it raise
# `RuntimeError: /dev/null is an empty file` at startup.
CMD ["python", "-m", "app.cli", "serve"]
