# `v12/scripts/` — CPU-side pipeline steps

Everything that runs on a laptop. The GPU work lives in `../notebooks/`; these
scripts prepare its inputs and consume its outputs. Run them from `v12/`
(`python3 scripts/<name>.py`), or via the phase targets in `../Makefile`.

## Data preparation

| Script | Does |
|---|---|
| `export_v11_splits.py` | copies the EXACT v1.1 splits into `../data/v11/` (digest-checked, never regenerated) |
| `build_generated.py` | Phase 2b — turns the hand-authored lines in `../data/gen/src/` into `gen_v12.jsonl` |
| `build_public_data.py` | Phase 2a — cleans downloaded public Hinglish corpora (train-only use) |
| `build_aug.py` | Phase 3 — meaning-preserving augmented variants of the train rows |
| `build_kd.py` | Phase 3 — attaches teacher-v2 logits to the KD training sets, after notebook 02 |

## Evaluation and export

| Script | Does |
|---|---|
| `v11_logits.py` | logits of the frozen v1.1 `full.npz` on every v1.2 split, so old and new systems are scored by one evaluator |
| `eval_teacher.py` | Phase 1 teacher scores → `../results/teacher_v11data.json` |
| `eval_systems.py` | Phase 5 — the single comparison table across every system |
| `export_student.py` | Phase 6 — exports a checkpoint to the torch-free deploy format, quantizes to INT8, verifies and calibrates |
| `latency.py` | Phase 6 CPU latency under the whitepaper's v1.1 benchmark protocol |

## Infrastructure

| Script | Does |
|---|---|
| `build_notebooks.py` | **generates** `../notebooks/*.ipynb` — edit this, not the notebooks |
| `build_bundle.py` | packs `../bundle.zip` (code, configs, JSONL, hash test) for upload |
| `notebook_dryrun.py` | executes a generated notebook locally on CPU to catch cell-level bugs before burning GPU time |
| `smoke_test.py` | a few steps of every training script on 32 rows → `../results/smoke_test.json` |
| `make_hash_fixtures.py` | regenerates `../tests/hash_fixtures.json` from the original v1.1 `model.py` |
| `hash_digest.py` | digest of hashed pieces over every exported v1.1 row, to compare across machines |

`make_hash_fixtures.py` is the one script that rewrites a test's ground truth.
Run it only when the v1.1 hashing itself legitimately changed, never to make a
failing `test_hashing.py` go green.
