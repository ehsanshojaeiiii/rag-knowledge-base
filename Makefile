.PHONY: run down dev test check format

run: ## Build and start Postgres + API in Docker
	docker compose up --build --detach --wait
	@echo "API: http://localhost:8000  Swagger UI: http://localhost:8000/docs"

down: ## Stop the stack
	docker compose down

dev: ## Run the API locally with auto-reload (Postgres in Docker)
	uv sync
	docker compose up --detach --wait db
	uv run alembic upgrade head
	uv run fastapi dev

test: ## Run tests
	uv run pytest

check: ## Lint, type-check and test (what CI runs)
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src tests
	uv run pytest

format: ## Fix formatting
	uv run ruff check --fix .
	uv run ruff format .
