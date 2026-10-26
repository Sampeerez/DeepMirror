# Developer shortcuts. Everyday use only needs ./start.sh or start.bat.
TORCH ?= cpu

.PHONY: run dev-backend dev-frontend install lint test test-backend test-frontend build release clean

run:
	uv run --extra $(TORCH) --extra neural-f0 voxclone

dev-backend:
	uv run --extra $(TORCH) uvicorn --factory backend.app:create_app --reload

dev-frontend:
	cd frontend && npm run dev

install:
	uv sync --extra $(TORCH) --extra neural-f0
	cd frontend && npm ci

lint:
	uv run ruff check backend tests
	uv run pyright
	cd frontend && npm run lint

test-backend:
	uv run pytest

test-frontend:
	cd frontend && npm test

test: test-backend test-frontend

build:
	cd frontend && npm run build

release: build
	uv run python scripts/package_release.py

clean:
	rm -rf frontend/node_modules frontend/dist .venv .pytest_cache .ruff_cache
