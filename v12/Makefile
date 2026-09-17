# v1.2 local tasks. Run from v12/:  make bundle | make smoke | make test-hash | make notebooks
PY    ?= ../.venv-v12/bin/python
PY11  ?= ../.venv/bin/python

.PHONY: export-splits fixtures test-hash smoke bundle notebooks all

export-splits:
	$(PY) scripts/export_v11_splits.py

fixtures:            # ground truth comes from the original v1.1 model.py
	$(PY11) scripts/make_hash_fixtures.py

test-hash:
	$(PY) tests/test_hashing.py
	$(PY11) tests/test_hashing.py

smoke:
	$(PY) scripts/smoke_test.py

notebooks:
	$(PY) scripts/build_notebooks.py

bundle: notebooks
	$(PY) scripts/build_bundle.py

all: export-splits test-hash smoke bundle
