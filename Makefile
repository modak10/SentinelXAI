# SentinelXAI — one-command entrypoints (MLOPS_CHECKLIST.md: "Pipeline should be
# executable with one command"). Run `make` or `make help` to list targets.
#
# Paths below assume Windows (venv/Scripts/...), matching this project's dev
# environment. On Linux/macOS, replace `venv/Scripts/` with `venv/bin/`.

.PHONY: help setup data eda test lint

help:
	@echo "make setup  - create venv and install dev dependencies"
	@echo "make data   - run the dataset pipeline (raw -> processed parquet + report)"
	@echo "make eda    - run exploratory data analysis on the train split"
	@echo "make test   - run the unit test suite"
	@echo "make lint   - run ruff over src/, scripts/, tests/"

setup:
	python -m venv venv
	venv/Scripts/pip install -r requirements-dev.txt
	venv/Scripts/pip install -e .

data:
	venv/Scripts/python scripts/build_dataset.py

eda:
	venv/Scripts/python scripts/run_eda.py

test:
	venv/Scripts/python -m pytest -v

lint:
	venv/Scripts/python -m ruff check src scripts tests
