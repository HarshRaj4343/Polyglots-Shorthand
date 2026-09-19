# The Polyglot's Shorthand

A reproducible classifier for Romanized Hindi-English (Hinglish) customer-support messages. It predicts customer **intent** and **sentiment** using Unicode-preserving preprocessing, hashed word/character features, shorthand aliases, emoji interactions, and calibrated softmax heads.

> **Important:** The datasets are synthetic and authored for this project. Reported scores are development results, not evidence of production-level generalization.

## What it predicts

**Intent:** `cancel_order`, `refund`, `track_order`, `not_received`, `damaged_item`, or `feedback`.

**Sentiment:** `negative`, `neutral`, or `positive`.

The model always returns one intent; there is no unknown, out-of-scope, or multi-intent class.

## Quick start

Python 3.11 or newer is recommended.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python solution.py predict "refund kab milega bhai"
```

Example output:

```json
{
  "intent": {"label": "refund", "confidence": 0.999881386756897, "review_required": false},
  "sentiment": {"label": "neutral", "confidence": 0.8672024607658386, "review_required": false}
}
```

Confidence is a calibrated probability estimate, not a correctness guarantee. The review flag is based on a validation-selected threshold.

### CLI commands

```sh
# One JSON result per input line
.venv/bin/python solution.py predict --stdin

# Evaluate the supplied full checkpoint
.venv/bin/python solution.py evaluate

# Run a warm latency benchmark
.venv/bin/python solution.py benchmark

# Rebuild data, train all configurations, evaluate, benchmark, and test
PYTHON=.venv/bin/python ./run.sh
```

## Streamlit app

The app supports single-message and batch inference, probability charts, review flags, attention visualization, model selection, and CSV export.

```sh
.venv/bin/python -m pip install -r requirements-app.txt
.venv/bin/streamlit run app.py
```

When available, the app defaults to the v1.2 distilled INT8 ONNX student in `v12/deploy/student_kd_s42/`. Otherwise it falls back to the v1.1 NumPy checkpoints in `artifacts/`.

## Architecture

1. Normalize text to NFC, lowercase it, collapse whitespace, and reject empty or overlong inputs.
2. Extract word unigrams/bigrams, character 3–5 grams, shorthand aliases, and symbol/emoji-to-word interactions.
3. Hash features into 32,768 CRC32 buckets and apply log scaling plus L2 normalization.
4. Run separate six-class intent and three-class sentiment heads.
5. Apply temperature calibration and confidence-based review thresholds.

## Repository layout

```text
.
├── app.py                    # Streamlit interface
├── model.py                  # Features, model, training, calibration, persistence
├── make_data.py              # Synthetic seeds, splitting, and augmentation
├── solution.py               # CLI, evaluation, ablations, and benchmarks
├── audit.py                  # Frozen post-development audit runner
├── audit.jsonl               # 24 additional synthetic audit examples
├── test_solution.py          # Implementation tests
├── run.sh                    # End-to-end reproduction script
├── artifacts/                # Trained checkpoints and reports
├── data/                     # Generated train/validation/test JSONL splits
├── v12/                      # Distillation and ONNX student subproject
├── WHITEPAPER.md             # Editable technical report
└── Q4_Solution.pdf           # Formatted technical report
```

### Checkpoints

| File | Description |
|---|---|
| `artifacts/full.npz` | Words, character n-grams, aliases, symbol interactions, and attention |
| `artifacts/char_word.npz` | Words, word pairs, and character n-grams |
| `artifacts/word_only.npz` | Words and word pairs only |
| `artifacts/strip_symbols.npz` | Full pipeline after removing symbols and punctuation |
| `v12/deploy/student_kd_s42/` | Distilled ONNX INT8 student used by the app when available |

## Reproduce and test

```sh
.venv/bin/python solution.py reproduce
.venv/bin/python audit.py
.venv/bin/python -m unittest -v test_solution.py
```

The current captured run in `artifacts/test_run.txt` reports 13 tests passing. The tests cover attention gradients and legacy loading, Unicode and emoji preservation, zero-width joiners, negation and symbol ablation, input and parameter limits, probability normalization, save/reload behavior, split integrity, and stress-test cues.

The reproduction pipeline generates group-separated splits, trains four v1.1 feature configurations, calibrates them, evaluates development and stress performance, checks emoji pairs, benchmarks inference, and writes reports under `artifacts/`.

## Evaluation snapshot

The frozen synthetic audit contains 24 examples created after the final model configuration and not used for training or calibration:

| Task | Accuracy | Macro-F1 | Coverage | Accepted accuracy |
|---|---:|---:|---:|---:|
| Intent | 87.5% | 0.865 | 91.7% | 95.5% |
| Sentiment | 87.5% | 0.806 | 87.5% | 95.2% |

See `artifacts/audit_results.json` for confusion matrices and individual errors. See `WHITEPAPER.md` and `artifacts/results.json` for the complete methodology and development metrics.

## Known limitations

- Training and audit data are synthetic, small, and single-author.
- Negation scope such as `cancel mat karna ... track` can still produce a highly confident wrong intent.
- Sentiment is weaker on sarcasm, mixed affect, complaints, and short praise.
- Unicode acceptance does not imply semantic coverage for other languages, dialects, or unseen slang.
- Emoji sequences are preserved at code-point level, but tokenization is not fully grapheme-cluster aware.
- Review thresholds are calibrated on small validation sets and do not guarantee accepted accuracy on new traffic.
- Local latency measurements exclude startup, model loading, networking, concurrency, and saturation.

For production use, collect consented independently labeled conversations, split by conversation/author/time, add contrast sets for negation and sarcasm, and measure calibration plus end-to-end latency under realistic load.

## Python usage

```python
from pathlib import Path
from model import Model

model = Model.load(Path("artifacts/full.npz"))
result = model.predict("item delivered bol raha hai but mila hi nahi 😒")
print(result["intent"]["label"])
print(result["sentiment"]["label"])
```

Only load trusted checkpoints. The loader uses `allow_pickle=False`, but deliberately oversized arrays can still consume resources.

## Documentation and references

- [`WHITEPAPER.md`](WHITEPAPER.md): architecture, data, evaluation, and limitations
- [`Q4_Solution.pdf`](Q4_Solution.pdf): formatted technical report
- [`artifacts/README.md`](artifacts/README.md): generated checkpoints and reports
- [`v12/README.md`](v12/README.md): distillation and deployment details

Background references:

1. [Bag of Tricks for Efficient Text Classification](https://aclanthology.org/E17-2068/)
2. [Feature Hashing for Large Scale Multitask Learning](https://arxiv.org/abs/0902.2206)
3. [Unicode Technical Standard #51](https://www.unicode.org/reports/tr51/)

This is an original NumPy baseline and does not import benchmark numbers from these references.
