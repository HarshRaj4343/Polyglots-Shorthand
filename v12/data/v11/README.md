# `v12/data/v11/` — the frozen v1.1 splits

An exact copy of the v1.1 data, written once by `../../scripts/export_v11_splits.py`
and never regenerated. v1.2 trains and reports on these same rows so that any
change in the numbers is attributable to the model, not to a reshuffled dataset.

| File | Rows | Groups | Use in v1.2 |
|---|---:|---:|---|
| `train.jsonl` | 536 | 130 | training |
| `validation.jsonl` | 158 | 39 | early stopping, selection, calibration — the only split tuned against |
| `test.jsonl` | 190 | 44 | reported only |
| `test_stress.jsonl` | 190 | 44 | `test` under the v1.1 14-entry stress map; robustness, reported only |
| `audit.jsonl` | 24 | 24 | the separately authored v1.1 audit; reported only |
| `MANIFEST.json` | — | — | SHA-256 of each file above |

## Do not regenerate these

`MANIFEST.json` digest-checks every file, and the digests also match
`dataset_sha256` in the repo root's `artifacts/results.json`. Re-running the
v1.1 generator would produce equivalent-looking data with different group
assignments and quietly break the comparison that the whole v1.2 result rests
on. Copy from `../../../data/` only through `export_v11_splits.py`, which
verifies as it writes.

One known wrinkle, documented rather than fixed: 5 of the 536 train rows have
char-3-gram Jaccard > 0.6 with a validation/test/audit message, because v1.1
split by paraphrase group rather than surface similarity. They are left in place
because the splits must be reused exactly. New v1.2 data *is* filtered against
those splits.
