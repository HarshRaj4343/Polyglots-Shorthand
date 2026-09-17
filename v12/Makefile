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

eval-teacher:
	$(PY) scripts/eval_teacher.py

teacher-v2:          # after unpacking notebook 02 outputs into runs/02_teacher_kd
	$(PY) scripts/eval_teacher.py runs/02_teacher_kd results/teacher_v2.json
	$(PY) scripts/build_kd.py

deploy:              # after unpacking notebook 03 outputs into runs/03_student
	$(PY) scripts/export_student.py --student runs/03_student/runs/student_kd_s42/student.pt --out deploy/student_kd_s42
	$(PY) scripts/latency.py deploy/student_kd_s42

compare:
	$(PY) scripts/eval_systems.py
