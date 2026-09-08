# ResumeIQ developer tasks. `make help` lists everything.
.DEFAULT_GOAL := help
.PHONY: help install dev api ui test test-fast cov lint fmt type check clean docker-build docker-up docker-down smoke

VENV    := .venv
ifeq ($(OS),Windows_NT)
PY      := $(VENV)/Scripts/python.exe
BIN     := $(VENV)/Scripts
else
PY      := $(VENV)/bin/python
BIN     := $(VENV)/bin
endif

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

install: ## Create the virtualenv and install everything
	python -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,ui]"

api: ## Run the API with autoreload
	$(PY) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

ui: ## Run the Streamlit dashboard
	$(PY) -m streamlit run ui/streamlit_app.py

test: ## Run the full test suite
	$(PY) -m pytest -q

test-fast: ## Run everything except the slower UI integration tests
	$(PY) -m pytest -q --ignore=tests/test_ui.py

cov: ## Run tests with a coverage report
	$(PY) -m pytest -q --cov=app --cov-report=term-missing --cov-report=html

lint: ## Check formatting and lint rules
	$(BIN)/ruff check app tests ui

fmt: ## Auto-fix what ruff can fix
	$(BIN)/ruff check app tests ui --fix

type: ## Static type check
	$(BIN)/mypy app

check: lint type test ## Everything CI runs

smoke: ## Score the bundled fixtures from the command line
	$(PY) -m app.cli analyze tests/fixtures/strong_resume.txt --job tests/fixtures/backend_job.txt -q

docker-build: ## Build both images
	docker compose build

docker-up: ## Start the stack (API on :8000, UI on :8501)
	docker compose up -d

docker-down: ## Stop the stack
	docker compose down

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build *.egg-info
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
