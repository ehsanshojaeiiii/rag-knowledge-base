.PHONY: run down dev test check format

run: ## Build and start Postgres, Ollama and the API in Docker
	docker compose up --build --detach --wait
	@echo "API: http://localhost:8000  Swagger UI: http://localhost:8000/docs"

down: ## Stop the stack
	docker compose down

dev: ## Run the API locally with auto-reload (Postgres and Ollama in Docker)
	uv sync
	docker compose up --detach --wait db ollama-pull
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
