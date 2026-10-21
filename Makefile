# FreshGuard AI task runner. CONFIG selects the dataset configuration.
CONFIG ?= configs/freshretailnet.yaml
PY ?= .venv/bin/python
FG = $(PY) -m freshguard.cli

.PHONY: help setup lint typecheck test smoke data eda forecast evaluate simulate report all dashboard clean-artifacts

help:
	@echo "setup | lint | typecheck | test | smoke | data | eda | forecast | evaluate | simulate | all | dashboard"

setup:            ## create .venv and install the package with dev tools
	python3 -m venv .venv && $(PY) -m pip install -U pip && $(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check src tests scripts dashboard

typecheck:
	$(PY) -m mypy

test:             ## unit + integration tests (synthetic data only)
	$(PY) -m pytest

smoke:            ## minimal end-to-end pipeline on the synthetic fixture
	$(FG) all --config configs/synthetic.yaml

data:
	$(FG) data --config $(CONFIG)
eda:
	$(FG) eda --config $(CONFIG)
forecast:
	$(FG) forecast --config $(CONFIG)
evaluate:
	$(FG) evaluate --config $(CONFIG)
simulate:
	$(FG) simulate --config $(CONFIG)
report:
	$(FG) report --config $(CONFIG)

all:              ## full experiment: data -> eda -> forecast -> evaluate -> simulate -> report
	$(FG) all --config $(CONFIG)

dashboard:
	FRESHGUARD_CONFIG=$(CONFIG) $(PY) -m streamlit run dashboard/app.py

clean-artifacts:  ## remove generated (git-ignored) artifacts for one config
	@echo "Delete data/processed/<name>, models/<name> manually after reviewing them."
