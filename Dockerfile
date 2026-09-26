# One image for all three services; docker-compose.yml picks the command.
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY minilab ./minilab
RUN uv sync --frozen --no-dev
EXPOSE 3000 8000 8001
