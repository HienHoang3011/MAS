UV=uv

.PHONY: setup sync run dev lint format type test index-knowledge hooks hooks-strict hooks-update clean

setup:
	$(UV) sync --extra dev
	$(UV) run --extra dev pre-commit install

sync:
	$(UV) sync --extra dev

run:
	$(UV) run uvicorn app.main:app --host 127.0.0.1 --port 2003 --reload

dev: run

lint:
	$(UV) run --extra dev ruff check app tests

format:
	$(UV) run --extra dev ruff check app tests --fix
	$(UV) run --extra dev ruff format app tests

type:
	$(UV) run --extra dev mypy app

test:
	$(UV) run --extra dev pytest

index-knowledge:
	$(UV) run python -m scripts.index_knowledge

# Auto-fix formatting (lenient mode)
hooks:
	$(UV) run --extra dev pre-commit run --all-files

# Strict checking (mypy, bandit, etc.)
hooks-strict:
	./scripts/check-strict.sh

hooks-update:
	$(UV) run pre-commit autoupdate
	$(UV) add --dev --upgrade pre-commit

clean:
	find . -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
