FROM python:3.13.18-slim-trixie AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-editable

FROM python:3.13.18-slim-trixie
RUN useradd --system --uid 10001 app
WORKDIR /app
COPY --from=build --chown=app:app /app/.venv .venv
COPY pyproject.toml alembic.ini ./
COPY migrations migrations
ENV PATH="/app/.venv/bin:$PATH"
USER app
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && fastapi run --host 0.0.0.0 --port 8000"]
