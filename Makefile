.PHONY: install run test lint format typecheck check

install:
	uv sync --all-groups

run:
	uv run orchard-serve

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy

check: lint typecheck test

