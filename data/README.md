# `data/` — v1.0/v1.1 dataset splits

The three JSONL splits the root pipeline trains and reports on. They are
**generated**, not hand-maintained: `make_data.py` expands 213 LLM-authored seeds
with rule-based augmentation, then splits by *paraphrase group* so that no
rewording of a message can straddle two splits. Regenerate with
`python3 solution.py reproduce`, which rewrites this folder from seed 20240615.

| File | Rows | Groups | Use |
|---|---:|---:|---|
| `train.jsonl` | 536 | 130 | fitting weights |
| `validation.jsonl` | 158 | 39 | early stopping, model selection, temperature calibration |
| `test.jsonl` | 190 | 44 | development test — reported, never tuned against |

One JSON object per line:

```json
{"id": "...", "text": "refund kab milega bhai", "intent": "refund", "sentiment": "neutral", "group_id": "..."}
```

`intent` is one of the six labels in `model.py:INTENTS`; `sentiment` is
negative / neutral / positive.

SHA-256 digests of all three files are recorded under `dataset_sha256` in
`artifacts/results.json`, so a regenerated copy can be proven identical. v1.2
reuses these exact splits rather than regenerating them — see `v12/data/v11/`.

The 24-row post-development audit set lives at the repo root as `audit.jsonl`,
deliberately outside this folder: it was authored separately, after development
froze, and is never part of the train/validation/test cycle.
