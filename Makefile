.PHONY: run run-gpu down dev test test-live check format

run: ## Build and start Postgres, Ollama and the API in Docker
	docker compose up --build --detach --wait
	@echo "Questions: http://localhost:8000  Swagger UI: http://localhost:8000/docs"

run-gpu: ## Same, with the language model in Docker Model Runner (Apple GPU on a Mac)
	docker compose -f compose.yaml -f compose.gpu.yaml up --build --detach --wait
	@echo "Questions: http://localhost:8000  Swagger UI: http://localhost:8000/docs"

down: ## Stop the stack (either variant)
	docker compose -f compose.yaml -f compose.gpu.yaml --profile ollama down

dev: ## Run the API locally with auto-reload (Postgres and Ollama in Docker)
	uv sync
	docker compose up --detach --wait db ollama-pull
	uv run alembic upgrade head
	uv run fastapi dev

test: ## Run tests
	uv run pytest

test-live: ## Real scenarios against the running stack (`make run` or `make run-gpu` first); slow, needs the API key
	LIVE_API=http://localhost:8000 uv run pytest tests/test_live.py -v

check: ## Lint, type-check and test (what CI runs)
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src tests
	uv run pytest

format: ## Fix formatting
	uv run ruff check --fix .
	uv run ruff format .
