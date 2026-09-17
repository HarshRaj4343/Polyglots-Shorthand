# v1.2 data card

All v1.2 additions go to TRAIN only. The v1.1 train/validation/test/audit splits and group ids are reused
exactly (see `v11/MANIFEST.json`; source digests match `artifacts/results.json`). Validation is the only split
used for early stopping, selection and calibration.

## v1.1 splits (`data/v11/`)
| File | Rows | Groups | Source | Use |
|---|---:|---:|---|---|
| train.jsonl | 536 | 130 | LLM-authored seeds + rule augmentation (v1.1 `make_data.py`) | training |
| validation.jsonl | 158 | 39 | same | early stopping, selection, calibration |
| test.jsonl | 190 | 44 | same | development test (report only) |
| test_stress.jsonl | 190 | 44 | test with the v1.1 14-entry stress map | robustness (report only) |
| audit.jsonl | 24 | 24 | separately authored v1.1 audit | report only |

Note: 5 of the 536 v1.1 train rows already have char-3-gram Jaccard > 0.6 with a validation/test/audit message
(v1.1 split by paraphrase group, not by surface similarity). They are left unchanged because the splits must
be reused exactly; new v1.2 data is filtered against validation/test/audit/contrast.

## Diagnostic contrast set (`data/contrast.jsonl`)
- 40 hand-written minimal pairs (80 rows), written in Phase 0 BEFORE any generated training data, never used
  for training, tuning, checkpointing or calibration.
- Phenomena (10 pairs each): `negation_scope`, `neutral_nahi`, `negated_negative`, `unseen_affect`.
- Max char-3-gram Jaccard to any v1.1 train/val/test/audit message: 0.37.
- All `unseen_affect` affect words (raahat, tang, sukoon, nirash, chinta, dukhi, chidh, fikar, khinn, pareshan,
  ...) are absent from the v1.1 training split (checked on normalized tokens).
- Caveat: written by the same LLM assistant that wrote most training data.

## Public data (Phase 2a)
To be filled: source, ID/URL, license, size, label mapping, use, load failures.

## Generated support data (Phase 2b)
To be filled.
