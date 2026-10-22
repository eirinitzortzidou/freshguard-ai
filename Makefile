# FreshGuard AI task runner. CONFIG selects the dataset configuration.
CONFIG ?= configs/freshretailnet.yaml
PY ?= .venv/bin/python
FG = $(PY) -m freshguard.cli

.PHONY: help setup setup-latest lint typecheck test smoke data eda forecast evaluate simulate report all nested reproduce check dashboard clean-artifacts

help:
	@echo "setup | lint | typecheck | test | check | smoke | data | eda | forecast | evaluate | simulate | nested | all | reproduce | dashboard"

setup:            ## create .venv with the exact locked versions used for the reported results
	python3 -m venv .venv && $(PY) -m pip install -U pip \
	  && $(PY) -m pip install -r requirements/lock.txt && $(PY) -m pip install -e . --no-deps

setup-latest:     ## create .venv with the newest versions allowed by pyproject.toml
	python3 -m venv .venv && $(PY) -m pip install -U pip && $(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check src tests scripts dashboard

typecheck:
	$(PY) -m mypy

test:             ## unit + integration tests (synthetic data only)
	$(PY) -m pytest

check: lint typecheck test   ## everything CI runs

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

nested:           ## nested rolling-origin evaluation + calibration study (after forecast)
	$(FG) nested --config $(CONFIG)

reproduce:        ## independent clean rerun in runs/reproduce, compared with committed metrics
	$(PY) scripts/reproduce.py --config $(CONFIG) --out runs/reproduce

all:              ## full experiment: data -> eda -> forecast -> evaluate -> simulate -> nested -> report
	$(FG) all --config $(CONFIG)

dashboard:
	FRESHGUARD_CONFIG=$(CONFIG) $(PY) -m streamlit run dashboard/app.py

clean-artifacts:  ## remove generated (git-ignored) artifacts for one config
	@echo "Delete data/processed/<name>, models/<name> manually after reviewing them."
