# Q4: The Polyglot's Shorthand

This is a complete, runnable reference solution for Question 4 of the CS/AI practice problem statements. It handles two tasks for Romanized Hindi-English customer-support messages:

1. **Intent classification** - determines what the customer wants.
2. **Sentiment classification** - estimates whether the expressed tone is negative, neutral, or positive.

The system uses Unicode-preserving preprocessing, word and character features, fixed-size feature hashing, and two linear softmax classifiers written with NumPy. It needs no GPU, API key, external model, or large ML framework.

This is a measured baseline. Intent is the stronger task. Sentiment, sarcasm, and long-distance negation remain difficult; the supplied reports document those failures.

## 1. Files in the submission

```text
Q4_Polyglots_Shorthand/
|-- README.md                    # This detailed guide
|-- Q4_Solution.pdf              # Formatted eight-page technical report
|-- WHITEPAPER.md                # Editable report text
|-- requirements.txt             # Required NumPy version
|-- run.sh                       # One-command reproduction
|-- model.py                     # Features, model, training, calibration, save/load
|-- make_data.py                 # Seed data, splitting, and augmentation
|-- solution.py                  # CLI, evaluation, ablations, stress test, benchmark
|-- audit.py                     # Separate post-development audit runner
|-- audit.jsonl                  # 24 additional audit examples
|-- test_solution.py             # 11 implementation tests
|-- data/
|   |-- train.jsonl              # Generated training split
|   |-- validation.jsonl         # Model-selection/calibration split
|   `-- test.jsonl               # Development test split
`-- artifacts/
    |-- full.npz                 # Main trained model
    |-- word_only.npz            # Word-only ablation
    |-- char_word.npz            # Word + character ablation
    |-- strip_symbols.npz        # Symbol-removal ablation
    |-- results.json             # Metrics, errors, timing, settings, hashes
    |-- audit_results.json       # Separate audit metrics and errors
    `-- test_run.txt             # Captured output from the tests
```

`SHA256SUMS.json` stores a SHA-256 digest for every packaged file. Python may create `__pycache__/`; it is disposable bytecode and not part of the solution.

## 2. Requirements and quick start

Requirements:

- Python 3.11 or newer
- NumPy 2.3.5, pinned in `requirements.txt`
- macOS, Linux, Windows, or a compatible Python environment

Create an isolated environment from this folder:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Try the supplied trained model:

```sh
.venv/bin/python solution.py predict "refund kab milega bhai"
```

Example output:

```json
{
  "intent": {
    "label": "refund",
    "confidence": 0.999881386756897,
    "review_required": false
  },
  "sentiment": {
    "label": "neutral",
    "confidence": 0.8672024607658386,
    "review_required": false
  }
}
```

`confidence` is the calibrated softmax probability of the selected label. It is not a correctness guarantee. `review_required` becomes true when confidence is below a threshold chosen on validation data.

Windows PowerShell paths are slightly different:

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe solution.py predict "refund kab milega bhai"
```

## 3. End-to-end flow

```text
Authored seed messages
        |
        v
Group-aware train / validation / test split
        |
        v
Spelling and shorthand variants within each split
        |
        v
Unicode normalization and token extraction
        |
        +--> word unigrams and bigrams
        +--> character 3-, 4-, and 5-grams
        +--> shorthand alias features
        `--> symbol/emoji-to-word interaction features
        |
        v
CRC32 hashing into 32,768 feature slots
        |
        v
Log-scaled, L2-normalized sparse vector
        |
        +--> 6-class intent softmax head
        `--> 3-class sentiment softmax head
        |
        v
Temperature calibration and review thresholds
        |
        v
JSON labels, confidence values, and review flags
```

Training and evaluation proceed as follows:

1. `make_data.py` creates three JSONL splits.
2. `solution.py` rejects group or exact-text leakage across splits.
3. `model.py` extracts features and learns both task heads.
4. Validation data chooses the best epoch, temperatures, and review thresholds.
5. `solution.py` evaluates the selected checkpoint on the development test.
6. It repeats training for three feature ablations.
7. It runs a spelling-stress test, emoji-pair check, and latency benchmark.
8. `audit.py` measures the final model on 24 additional examples.
9. `test_solution.py` verifies implementation invariants.

## 4. Labels

### Intent labels

| Label | Meaning | Example |
|---|---|---|
| `cancel_order` | Cancel an order or booking | `order cancel krdo please` |
| `refund` | Ask for money back or refund progress | `refund kab tak aayega` |
| `track_order` | Ask about shipment timing/location | `parcel kidhar pahuncha` |
| `not_received` | Marked delivered but not received | `delivered dikh raha but mila nahi` |
| `damaged_item` | Broken, cracked, leaking, or damaged goods | `bottle leak ho rahi hai` |
| `feedback` | General service praise, criticism, or feedback | `service bahut acchi thi` |

The model returns one intent. It has no explicit unknown or multi-intent label.

### Sentiment labels

| Label | Annotation rule |
|---|---|
| `negative` | Clear anger, dissatisfaction, frustration, or constructed sarcasm |
| `neutral` | Mainly a factual request with no clear expressed emotion |
| `positive` | Clear praise, gratitude, satisfaction, or delight |

These labels describe expressed emotion. A refund request is not automatically negative.

## 5. JSONL data format

Every line in a split file is an independent JSON object:

```json
{
  "text": "bhai refund kab tak aayega",
  "intent": "refund",
  "sentiment": "neutral",
  "group": "refund-00",
  "split": "train",
  "language": "hi-en",
  "source": "authored_synthetic"
}
```

- `text`: exact model input.
- `intent`: one of six intent labels.
- `sentiment`: one of three sentiment labels.
- `group`: original seed identity; variants share this ID.
- `split`: train, validation, or test.
- `language`: Hindi-English Romanization in this dataset.
- `source`: confirms the message is authored synthetic data.

JSONL means one JSON value per physical line. Do not wrap the file in a JSON array unless `read_data()` is also changed.

## 6. `make_data.py`, chunk by chunk

This file owns data authorship, split assignment, and controlled augmentation.

### Imports

```python
import json
import random
from pathlib import Path
```

`json` writes readable JSONL, `random` provides deterministic shuffling, and `Path` handles file paths.

### `SEEDS`

`SEEDS` is keyed by intent. Every value contains 16 manually authored messages. A tuple such as:

```python
('u', 'bhai refund kab tak aayega')
```

contains a sentiment code and text. Codes are `n` for negative, `u` for neutral, and `p` for positive. There are 96 base seeds total. No customer data or downloaded corpus is used.

Most positive examples belong to `feedback`, creating an intent-sentiment correlation. This helps explain why sentiment generalizes poorly.

### `SHORT`

`SHORT` maps full spellings to common shorthand for augmentation, such as `kar -> kr`, `nahi -> nhi`, and `please -> plz`. It never removes negation or an emoji.

### `shorthand(text)`

```python
return ' '.join(SHORT.get(t, t) for t in text.split())
```

It splits on whitespace, replaces known tokens, preserves unknown ones, and rejoins with one space. It is intentionally simple and does not handle punctuation attached to a word.

### `variants(text)`

This produces up to three variants:

1. Original text.
2. The `shorthand()` form.
3. Missing-vowel spellings such as `order -> ordr`, `delivery -> delivry`, and `service -> servis`.

`dict.fromkeys()` removes duplicates while preserving order. The augmenter avoids adding/removing negation, emoji, or sentiment punctuation because those changes could invalidate the label.

### `build(root)`

`random.Random(42)` creates a local, repeatable random generator. Each intent is handled separately. Within an intent, seed IDs are grouped by sentiment and shuffled, so available sentiment classes are distributed across splits.

Approximately 18% of eligible seed groups go to validation and 22% to test; the rest go to training. A sentiment with at least three seeds contributes at least one seed to validation and one to test.

Each seed gets a group ID such as `refund-04`. Split assignment happens **before augmentation**, and every derived spelling stays in the seed's group and split. This prevents a near-duplicate from appearing in both training and test.

Positive feedback seeds create two extra constructed forms:

```text
positive praise + 😊 -> positive
positive praise + 😒 -> negative
```

This is a synthetic annotation assumption for those pairs, not a universal emoji rule.

The nested loops add full records to `rows`, then write one file per split with `ensure_ascii=False` so emoji remain readable.

When run directly, the final block calls `build()` and prints split row counts.

## 7. `model.py`, chunk by chunk

This file implements normalization, feature extraction, training, calibration, prediction, and model persistence.

### Constants and schema

`INTENTS` and `SENTIMENTS` define label order. Intent occupies model columns 0-5 and sentiment columns 6-8. Reordering labels without retraining would attach weights to the wrong names.

`DIM = 32768` fixes the number of hash buckets. `MAX_CHARS = 512` bounds request work and prevents accidentally treating a long chat as one short message.

`ALIASES` maps shorthand toward canonical forms during inference, such as `kr -> kar`, `nhi -> nahi`, and `bht -> bahut`. Raw features remain present, so normalization is additive rather than destructive.

### Token pattern

```python
TOKEN = re.compile(r'\w+|[^\w\s]', re.UNICODE)
```

It matches either a run of Unicode word characters or one non-word, non-space character. Words stay grouped; punctuation and many emoji code points remain visible. This is not full grapheme segmentation, so a complex joined emoji may become multiple interaction tokens.

### `normalize(text)`

This function:

1. Rejects non-string values.
2. Rejects inputs over 512 Unicode code points.
3. normalizes canonically equivalent sequences to NFC.
4. lowercases text.
5. collapses whitespace runs.
6. rejects empty/whitespace-only input.

Lowercasing makes capitalization invariant but loses emphasis such as `VERY BAD`.

### `features(text, mode='full')`

This returns sparse active indices (`int32`) and their normalized values (`float32`). It never creates a dense input vector.

In `strip_symbols` mode, Unicode categories beginning with `S` or `P` are removed. This is an ablation to measure the effect of discarding symbols and punctuation.

All modes create word unigrams prefixed `w:` and adjacent bigrams prefixed `b:`. For `refund nahi mila`, features include `w:refund`, `b:refund|nahi`, and `b:nahi|mila`.

Every mode except `word_only` pads the whole message with `^` and `$`, then extracts overlapping character 3-, 4-, and 5-grams. Similar spellings share pieces, helping the model handle unseen typos.

`full` and `strip_symbols` also add canonical alias unigrams (`a:`) and bigrams (`ab:`). An alias can expand, for example `krdo -> kar do`.

They also create `e:` interactions between up to eight distinct symbol tokens and 64 distinct word tokens. For `badhiya service 😒`, one feature resembles `e:😒|badhiya`. The emoji meaning is learned from labels; no emoji sentiment is hard-coded.

Feature strings are mapped with:

```python
zlib.crc32(feature.encode('utf-8')) % 32768
```

Hashing fixes memory use and removes the need to store a vocabulary. Different strings can collide. `Counter` combines repeats/collisions. A count becomes `1 + ln(count)`, reducing the effect of repetition. Values are then L2-normalized so their squared sum is approximately one.

### `softmax(z)`

Softmax converts scores into probabilities. Subtracting `max(z)` before exponentiation prevents numerical overflow without changing the mathematical result.

### `Model.__init__()`

The class initializes:

- a `32768 x 9` float32 weight matrix;
- nine float32 biases;
- two temperatures;
- two review thresholds;
- the chosen feature mode.

The learned arrays contain `32768 * 9 + 9 = 294,921` parameters, far below 500 million.

### `probabilities(text)` and `predict(text)`

`probabilities()` extracts features and computes `v @ w[ix] + b`. It separately temperature-scales and softmaxes the first six intent logits and final three sentiment logits.

`predict()` selects each head's highest probability, converts its index to a label, and compares confidence against the review threshold. It returns a JSON-friendly dictionary.

### `fit(train, valid, epochs=55, seed=42)`

Training features and integer targets are precomputed. Separate class weights use:

```text
N / (number_of_classes * class_count)
```

Rare labels therefore produce larger gradients.

For 55 epochs, the learning rate is `0.6 / (1 + epoch/20)`. The matrix is multiplied by `0.999` once per epoch for simple weight decay. Examples are shuffled with a seeded NumPy generator.

For each example, the code computes two softmax vectors, subtracts one at each true class, applies class weights, updates only active feature rows, and updates biases at one tenth of the weight learning rate. This is sparse, example-level stochastic gradient descent on two cross-entropy losses.

After every epoch, `nll(valid)` measures validation negative log likelihood. The code retains a copy of the best weights, biases, and epoch, then restores that checkpoint after training.

### `nll(rows)`

For both tasks and every row, it takes negative log probability of the true label and averages all terms. Probabilities are clipped to at least `1e-9` before `log` to avoid infinity.

### `calibrate(valid)`

Calibration stores validation logits and tests 30 log-spaced temperatures from 0.4 to 4.0 per task. The temperature with lowest validation negative log likelihood is selected. Temperature changes confidence sharpness but not the predicted class.

It then tries every distinct validation confidence as a review cutoff. A threshold qualifies when it accepts at least five examples with at least 90% observed accuracy. It chooses maximum coverage; if none qualify, threshold `1.01` flags everything.

This small-set policy is illustrative. It does not guarantee 90% accuracy on new data.

### `save(path)` and `load(path)`

`save()` writes compressed NumPy arrays `w` and `b`, plus JSON metadata containing feature mode, temperatures, thresholds, epoch, dimension, maximum input length, and label lists.

`load()` uses `allow_pickle=False`, checks dimension and exact label order, copies arrays, restores metadata, and returns a ready model. Schema mismatch raises `ValueError`.

## 8. `solution.py`, chunk by chunk

This is the CLI and experiment orchestrator.

### Path setup and `read_data(split)`

`ROOT` is based on the script's own location, and `ART` points to its `artifacts/` directory. Commands therefore work even when launched from another current directory.

`read_data()` reads `data/<split>.jsonl`, ignores blank lines, and parses each line. Malformed JSON raises an error instead of being silently skipped.

### `validate_data(parts)`

Before training, this function:

1. Rejects group overlap across train, validation, and test.
2. Rejects exact lowercased text overlap across splits.
3. Validates every label against the model schema.
4. runs every text through feature extraction to catch invalid input.
5. Ensures every split contains all six intents.

It does not detect arbitrary semantic paraphrase leakage. Seed group IDs provide the main protection for generated variants.

### `summarize(y, p, labels, conf=None, threshold=0)`

This constructs a confusion matrix and calculates metrics without scikit-learn:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2TP / (2TP + FP + FN)
```

`macro_f1` is the mean of all class F1 scores, giving rare and frequent classes equal weight. When confidence is provided, the function also reports the fraction above threshold (`coverage`) and accuracy among those accepted predictions (`accepted_accuracy`).

### `evaluate(model, rows)`

This predicts every row and runs `summarize()` separately for intent and sentiment. It also saves every example where either task is wrong, including text, group, expected labels, and full predictions. These error records support qualitative analysis.

### `stress_text(text)`

This applies spelling changes that differ from training augmentation:

```text
refund -> refnd       order -> orrder
nahi   -> nahii       please -> pleez
delivery -> delivary  service -> serrvice
```

Negation and emoji remain present. This is a narrow synthetic stress test, not proof of universal typo invariance.

### `stress_evaluate(model, rows)`

It copies test rows with stress-transformed text and evaluates them. It also reports:

- `changed_fraction`: how many messages were actually modified.
- `prediction_consistency`: how often the new prediction equals the original prediction.

Consistency can be high while both predictions are wrong, so it must be read with F1 and accuracy.

### `emoji_pairs(model, rows)`

This finds exact held-out pairs where a positive praise message has an otherwise identical trailing-`😒` negative version. It records both predictions and counts a pair as correct only when both sides are correct.

The current test has five text variants from only two independent seed groups. The result must be interpreted at that small scale.

### `benchmark(model, rows, repeats=600)`

The nested measurement function performs 80 warm-ups followed by 600 serial timed predictions. Each timing includes normalization, feature extraction, both classifier heads, and JSON serialization.

It reports p50, p95, p99 milliseconds and aggregate messages per second for:

- development-test messages;
- a repeated synthetic sentence cut to 32, 128, and 512 code points.

It excludes process startup, model loading, network transport, queues, service frameworks, and concurrent saturation. Repeated text has fewer unique features than arbitrary text, so this is not a worst-case guarantee.

### `reproduce()`

The full experiment does the following:

1. Creates `artifacts/`.
2. Rebuilds the data splits.
3. Loads and validates data.
4. Records row/group/label counts.
5. Trains, calibrates, saves, and evaluates four feature modes.
6. Runs development, stress, and emoji-pair evaluations.
7. Benchmarks the saved full model.
8. Records parameter count, array bytes, compressed file size, environment metadata, and data hashes.
9. Writes `artifacts/results.json`.

The four modes are:

| Mode | Features | Purpose |
|---|---|---|
| `full` | words, bigrams, characters, aliases, symbol interactions | Main proposal |
| `word_only` | words and bigrams | Measures value of subword features |
| `char_word` | words, bigrams, character n-grams | Measures value of aliases/interactions |
| `strip_symbols` | full pipeline after removing symbols/punctuation | Measures value of retaining those cues |

Every ablation is retrained. Removing features only at inference would be an unfair comparison.

### `main()` and every CLI command

`argparse` defines five commands.

#### Full reproduction

```sh
python solution.py reproduce
```

Regenerates data, trains all modes, evaluates, benchmarks, and overwrites generated model/result files.

#### Single prediction

```sh
python solution.py predict "mera parcel kaha hai"
```

Loads `artifacts/full.npz` and predicts the supplied argument.

#### Persistent line-by-line prediction

```sh
python solution.py predict --stdin
```

Each input line produces one JSON line. Keeping the process alive avoids repeated model loading. Blank or oversized lines produce error JSON. End with Ctrl-D on macOS/Linux or Ctrl-Z then Enter in a Windows console.

#### Train one model

```sh
python solution.py train --mode full
python solution.py train --mode char_word
```

Uses existing generated splits, validates them, trains one mode, calibrates it, and overwrites `artifacts/<mode>.npz`. It does not regenerate data or update `results.json`.

#### Evaluate the full model

```sh
python solution.py evaluate
```

Loads `full.npz`, evaluates `data/test.jsonl`, and prints JSON without writing a file.

#### Benchmark the full model

```sh
python solution.py benchmark
```

Runs a fresh warm benchmark and prints JSON. Use `reproduce` when you want the result captured inside `results.json`.

## 9. `audit.py`, chunk by chunk

The audit is separate from development evaluation.

It reads 24 objects from `audit.jsonl`, gives each an in-memory group ID because `evaluate()` expects one for error records, loads the already-trained `full.npz`, and calls `evaluate()`.

It never calls `fit()` or `calibrate()`, so audit examples cannot change the model. It adds a protocol note, writes full results to `artifacts/audit_results.json`, and prints compact accuracy/macro-F1 output.

The audit was authored after the final feature configuration and was not used for more tuning. It remains small, synthetic, single-author data containing some prompt paraphrases, not an external benchmark.

## 10. `test_solution.py`, test by test

These are implementation tests, not model-quality tests.

| Test | What it verifies |
|---|---|
| `test_split_integrity` | Split leakage and schema checks run successfully |
| `test_emoji_preserved` | Normalization retains emoji and emoji changes full features |
| `test_symbol_ablation` | Symbol stripping makes text with/without `😒` identical |
| `test_negation_preserved` | `nahi` remains and changes the feature vector |
| `test_case_and_whitespace` | Case/spacing variants normalize identically |
| `test_unicode_and_zwj` | Joined emoji are preserved; several scripts produce normalized features |
| `test_input_limits` | Invalid types, empty input, and >512 code points fail correctly |
| `test_parameter_limit` | Learned arrays stay below 500 million parameters |
| `test_save_reload` | Saving and reloading preserves prediction output |
| `test_probabilities` | Both probability vectors are finite and sum to one |
| `test_stress_preserves_semantics_cues` | Stress transformation keeps negation and emoji/punctuation |

`TemporaryDirectory` makes the save/reload check self-cleaning. Several tests load the supplied full checkpoint.

Run tests with:

```sh
python -m unittest -v test_solution.py
```

A successful run ends with `Ran 11 tests` and `OK`.

## 11. `run.sh`, line by line

```sh
#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
```

- The shebang selects a standard POSIX shell.
- `set -e` stops at the first failure.
- `set -u` rejects undefined variables.
- `cd` moves to the package directory.
- `PYTHON` accepts a caller-supplied interpreter or defaults to `python3`.

The script then runs `solution.py reproduce`, `audit.py`, and the 11 tests in sequence.

Use the virtual environment explicitly:

```sh
PYTHON=.venv/bin/python ./run.sh
```

It overwrites generated data, checkpoints, and result JSON. It does not rewrite the README, whitepaper, or PDF.

## 12. `requirements.txt`

```text
numpy==2.3.5
```

NumPy provides arrays, matrix operations, seeded shuffling, percentiles, and compressed checkpoints. All other imports are from Python's standard library. There is no scikit-learn, PyTorch, TensorFlow, pandas, or tokenizer dependency.

## 13. Model artifacts and Python integration

The `.npz` files are compressed NumPy archives. Each contains `w`, `b`, and JSON `meta`.

| File | Feature mode |
|---|---|
| `full.npz` | Main model used by normal prediction |
| `word_only.npz` | Word/bigram ablation |
| `char_word.npz` | Word/bigram/character ablation |
| `strip_symbols.npz` | Symbol-removal ablation |

The full checkpoint is about 106 KB compressed. The weight and bias arrays occupy about 1.18 MB after loading. Binary checkpoint files should not be edited in a text editor.

Use the model in another Python program:

```python
from pathlib import Path
from model import Model

model = Model.load(Path('artifacts/full.npz'))
result = model.predict('item delivered bol rha h but mila hi nhi 😒')
print(result['intent']['label'])
print(result['sentiment']['label'])
```

Load only trusted checkpoints. `allow_pickle=False` blocks Python pickle objects, though deliberately huge arrays could still consume resources.

## 14. Understanding `results.json`

| Top-level field | Meaning |
|---|---|
| `seed` | Reproducibility seed |
| `data` | Row, group, and label counts for each split |
| `models` | All four models and their complete evaluations |
| `benchmark` | Warm latency/throughput results |
| `parameter_count` | Learned weight and bias count |
| `weight_bytes` | Array bytes, excluding Python/NumPy overhead |
| `saved_model_bytes` | Compressed `full.npz` size |
| `environment` | Python, NumPy, OS, release, architecture, CPU string |
| `dataset_sha256` | Digest of each generated split |

Every model entry contains training duration, selected epoch, temperatures, thresholds, validation metrics, development-test metrics, stress metrics, emoji-pair results, and detailed errors.

Confusion-matrix rows are true labels and columns are predicted labels. Always use the adjacent `label_order`.

`coverage` is the fraction above the review threshold. `accepted_accuracy` covers only that subset. High accepted accuracy with low coverage means the model handled a small easy fraction, not that overall quality is high.

## 15. Captured results

| Metric | Result |
|---|---:|
| Learned weights and biases | 294,921 |
| Float32 array storage | 1,179,684 bytes |
| Compressed full checkpoint | 105,953 bytes |
| Development test | 58 rows from 22 seed groups |
| Full intent accuracy / macro-F1 | 81.0% / 0.820 |
| Full sentiment accuracy / macro-F1 | 60.3% / 0.423 |
| Spelling-stress intent macro-F1 | 0.717 |
| Frozen synthetic audit | 24 messages |
| Audit intent accuracy / macro-F1 | 95.8% / 0.958 |
| Audit sentiment accuracy / macro-F1 | 91.7% / 0.831 |
| Warm development-message p95 | 0.218 ms |
| Warm repeated 512-code-point p95 | 0.847 ms |

Timing environment: Darwin 25.6.0, arm64, Python 3.12.14, NumPy 2.3.5. The exact CPU model was unavailable. Timings exclude loading, startup, transport, queues, framework overhead, and concurrency.

The development test influenced code fixes and is not a blind benchmark. The later audit was not used for more tuning but is also small and synthetic. Their score difference demonstrates sampling sensitivity.

## 16. Known failures and limitations

- Positive sentiment recall is 0/10 on the development test.
- Both labels are correct on 0/5 held-out praise-versus-`😒` pairs; these come from only two seed groups.
- `cancel mat karna sirf order track karke batao` is incorrectly classified as cancellation with about 0.996 confidence. The model retains negation but does not reliably understand its scope.
- `bohot badhiya service` is incorrectly negative in the audit, so the motivating sarcasm contrast is not reliably solved.
- Temperature scaling adjusts probability sharpness but cannot fix systematic errors.
- The full configuration does not consistently beat `char_word`.
- Unicode acceptance does not prove understanding beyond the tested Hindi-English data.
- Complex emoji are preserved at code-point level, but symbol tokenization is not fully grapheme-aware.
- Labels come from one author; there is no inter-annotator agreement.
- There is no unknown, out-of-scope, or multi-intent class.

## 17. Adding training examples

Edit `SEEDS` in `make_data.py`; generated JSONL files are overwritten by the next run.

1. Select the correct intent list.
2. Add `(sentiment_code, text)`.
3. Label expressed affect rather than operational severity.
4. Keep meaning-changing variants, especially negation/sarcasm changes, as separately labeled examples.
5. Run the full reproduction.
6. Inspect per-class metrics and individual errors.

Example:

```python
('n', 'refund ka wait karke thak gaya bahut bekar')
```

Then run:

```sh
PYTHON=.venv/bin/python ./run.sh
```

For a real deployment, replace synthetic seeds with consented, independently labeled conversations, and split by conversation/author/time before augmentation.

## 18. Adding or renaming labels

This requires retraining. Update `INTENTS` or `SENTIMENTS`, then change the current hard-coded dimensions and slices:

- weight and bias size `9`;
- intent boundary `6`;
- sentiment offset `6 + b`;
- calibration ranges `(0, 6)` and `(6, 9)`;
- class-balance dimensions.

Add enough data to every split, update tests/documentation, remove or rename incompatible old checkpoints, and retrain all modes. A production refactor should derive all dimensions from `len(INTENTS)` and `len(SENTIMENTS)`.

## 19. Changing features

### Hash size

Change `DIM`, then retrain. Existing models will fail the schema check. Approximate weight memory is `DIM * 9 * 4` bytes. Larger size reduces expected collisions but grows memory linearly.

### Shorthand

Extend `ALIASES` for canonical inference features and `SHORT` for generated variants. Keep raw features and test ambiguous mappings.

### Character lengths

Modify `(3, 4, 5)` in `features()`. Short n-grams overlap more but collide and carry less context; long n-grams are more precise but less typo-tolerant.

### New ablation

Implement it in `features()`, add it to CLI choices and the reproduction loop, write a targeted test, retrain, and document the result.

## 20. Recommended next iteration

1. Collect larger, consented, independently annotated data.
2. Balance sentiment inside every intent.
3. Add contrast sets for negation scope, multiple intents, emoji ambiguity, punctuation, dialects, and switch points.
4. Split by conversation, author, time, and paraphrase family.
5. Keep a blind test untouched until model and thresholds are frozen.
6. Compare this baseline with a compact contextual encoder.
7. Distill and quantize any larger model, counting all parameters.
8. Add unknown/out-of-scope and multi-intent detection.
9. Measure end-to-end p95/p99 under realistic concurrent traffic.

Keep this sparse model as a transparent speed baseline when testing a neural replacement.

## 21. Troubleshooting

### NumPy is missing

```sh
.venv/bin/python -m pip install -r requirements.txt
```

Use the same interpreter for installation and execution.

### `artifacts/full.npz` is missing

```sh
python make_data.py
python solution.py train --mode full
```

Or run the complete pipeline.

### `incompatible model schema`

The current dimension or labels differ from checkpoint metadata. Retrain after changing `DIM`, `INTENTS`, or `SENTIMENTS`.

### Input exceeds 512 code points

Split a conversation into messages or shorter windows upstream. Do not silently truncate the end because the final negation or emoji may change meaning.

### A confident prediction is wrong

Confidence is not a guarantee. Check known failures and training coverage. Add an independently labeled contrast family and retrain; avoid one-off output rules.

### Results changed after rerunning

Confirm NumPy version, split hashes, selected epochs, and temperatures. Floating-point details can vary across platforms; latency is expected to vary.

### `Permission denied: ./run.sh`

```sh
chmod +x run.sh
```

or:

```sh
PYTHON=.venv/bin/python sh run.sh
```

### JSONL parsing error

Every nonblank line must be valid standalone JSON with double-quoted strings, no comments, and no trailing comma.

## 22. Integrity verification

The delivered manifest describes the original packaged files. Check it with:

```sh
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path('.')
manifest = json.loads((root / 'SHA256SUMS.json').read_text())
for relative, expected in manifest.items():
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    print('OK' if actual == expected else 'CHANGED', relative)
PY
```

After intentionally editing source or this README, the prior manifest naturally reports `CHANGED`. Regenerate it only while preparing a new trusted release.

## 23. Which document to read

- `README.md`: setup, commands, and detailed code walkthrough.
- `Q4_Solution.pdf`: polished architecture, evaluation, trade-offs, and problem response.
- `WHITEPAPER.md`: editable technical report.
- `artifacts/results.json`: exact metrics, matrices, thresholds, timings, and errors.
- Python source: authoritative execution details.

## 24. References

1. Joulin et al., *Bag of Tricks for Efficient Text Classification*: https://aclanthology.org/E17-2068/
2. Weinberger et al., *Feature Hashing for Large Scale Multitask Learning*: https://arxiv.org/abs/0902.2206
3. Unicode Consortium, *Unicode Technical Standard #51*: https://www.unicode.org/reports/tr51/

This is an original NumPy baseline, not a reproduction of fastText. No benchmark number is imported from these sources.

## 25. Complete run checklist

```sh
cd Q4_Polyglots_Shorthand
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python solution.py predict "order cancel krdo please"
PYTHON=.venv/bin/python ./run.sh
.venv/bin/python solution.py evaluate
.venv/bin/python solution.py benchmark
```

After reproduction, confirm the tests end in `OK`, inspect `artifacts/results.json`, and read the documented errors before presenting the system as robust.
