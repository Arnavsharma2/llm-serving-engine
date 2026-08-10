.PHONY: install test lint benchmark-smoke serve docker-up

install:
	uv sync --extra all

test:
	uv run pytest

lint:
	uv run ruff check .

benchmark-smoke:
	uv run llmserve benchmark --config continuous --requests 8 --arrival-rate 20 --output artifacts/smoke.json

serve:
	uv run llmserve serve

docker-up:
	docker compose up --build
