# Inclination Score. On Windows without GNU make, use `.\make.ps1 <target>` (same targets).

ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
else
PY := .venv/bin/python
endif

# Package and model caches stay inside the project folder.
export UV_CACHE_DIR := .cache/uv

.PHONY: setup download inspect data train train-v3 llm export export-v2 eval app api test all

setup:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements.txt -e .

download:
	$(PY) -m inclination.load.download

inspect:
	$(PY) -m inclination.load.inspect_raw

data: download
	$(PY) -m inclination.load.standardise

train:
	$(PY) -m inclination.train

# Optional, slow (about 2 hours on a 4 GB GPU). Needs requirements-ml.txt. `export` picks the
# v3 models up automatically once they exist.
train-v3:
	$(PY) -m inclination.train --model v3

# Optional, costs money: Gemini cue tags, v2 rubric scores and rationales. Needs GEMINI_API_KEY
# in .env. Afterwards `export-v2` adds them to the dashboard files without redoing v1 and v3.
llm:
	$(PY) -m inclination.llm_jobs all

export-v2:
	$(PY) -m inclination.export --models v2 --append

export:
	$(PY) -m inclination.export

eval:
	$(PY) eval/evaluate.py

app:
	$(PY) -m streamlit run app/Home.py

# Data server for the React UI in ../web (reads outputs/ only).
api:
	$(PY) -m uvicorn api.server:app --port 8000 --reload

test:
	$(PY) -m pytest -q

all: data train export eval
