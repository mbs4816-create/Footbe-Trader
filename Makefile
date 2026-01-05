.PHONY: help install dev lint format test run check clean

help:
	@echo "Available commands:"
	@echo "  install  - Install production dependencies"
	@echo "  dev      - Install development dependencies"
	@echo "  lint     - Run linting checks"
	@echo "  format   - Format code"
	@echo "  test     - Run tests"
	@echo "  run      - Run heartbeat with dev config"
	@echo "  check    - Run all checks (lint + test)"
	@echo "  clean    - Remove build artifacts"

install:
	uv pip install -e .

dev:
	uv pip install -e ".[dev]"

lint:
	ruff check src tests
	mypy src

format:
	ruff check --fix src tests
	ruff format src tests

test:
	pytest tests/ -v --cov=src/footbe_trader --cov-report=term-missing

run:
	python scripts/run_heartbeat.py --config configs/dev.yaml

check: lint test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	rm -rf dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f data/*.db
