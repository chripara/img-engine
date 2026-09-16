# syntax=docker/dockerfile:1
FROM python:3.11

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .
RUN sed -i 's/\r$//' docker-entrypoint.sh && chmod +x docker-entrypoint.sh

ARG RUN_TESTS=false
RUN if [ "$RUN_TESTS" = "true" ]; then pytest tests/ -v; fi

ENV OLLAMA_URL=http://localhost:11434

EXPOSE 5000 7860

ENTRYPOINT ["/bin/bash", "./docker-entrypoint.sh"]
