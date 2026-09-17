# v1.2 progress

## Phase 0 - local setup (done)
- Exact v1.1 splits exported (digests match results.json) + test_stress; hashing copied into `polyglot12/` and verified byte-identical to v1.1 on 50 fixtures and all 1,148 rows under Python 3.10/3.12/3.14 (Unicode 13/15/16).
- Teacher and student trainers written with per-epoch checkpoints and auto-resume; CPU smoke test (32 rows, simulated disconnect + resume, all 4 teacher tokenizers, 3 student variants, augmentation-safety check) passes -> `results/smoke_test.json`; notebook 01 dry-run passes.
- Finding: hing-bert and MuRIL tokenize every emoji as [UNK] (261/908 rows affected); only the hing-roberta models can see emoji. Contrast set (40 pairs) written early so notebook 01 scores it.
