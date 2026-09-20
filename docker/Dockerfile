FROM ghcr.io/astral-sh/uv:latest AS uv_bin

FROM python:3.12-slim AS builder
COPY --from=uv_bin /uv /bin/uv
WORKDIR /build

COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install into /install using uv with pytorch-cpu wheel resolution
RUN uv pip install --system --prefix=/install --no-cache .

FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 1000 appuser
COPY --from=builder /install /usr/local
COPY --chown=appuser:appuser models/ /app/models/
WORKDIR /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "prodml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
