# syntax=docker/dockerfile:1.7
# ---- build: compile wheels for the app and all dependencies -----------------------------
FROM python:3.12-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir --upgrade pip
COPY pyproject.toml ./
COPY src ./src
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

# ---- runtime: no compilers, non-root, read-only friendly ----------------------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    RAG_DOCS_DIR=/app/data/docs \
    RAG_STATIC_DIR=/app/static \
    RAG_CHROMA_DIR=/tmp/chroma
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
COPY data ./data
COPY static ./static
USER 10001
EXPOSE 8080
# One worker per container: scale with replicas, not workers (limits are global via Redis).
# Graceful shutdown leaves running SSE streams up to 60s to finish (match terminationGracePeriod).
CMD ["uvicorn", "northwind_rag.api:app", "--host", "0.0.0.0", "--port", "8080", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--timeout-graceful-shutdown", "60", \
     "--no-server-header"]
