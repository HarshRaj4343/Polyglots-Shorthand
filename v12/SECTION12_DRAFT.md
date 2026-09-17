# Section 12 (draft): Version 1.2 — Teacher Distillation into a Hashed-Piece Transformer

> Status: DRAFT. Every number comes from `v12/results/*.json`. Items marked **[PENDING 03]** are filled in
> after notebook 03 (student training) returns; nothing below them is estimated.

## 12.1 Motivation

Version 1.1 left four documented failures (Section 11): unseen affect words (*pareshan*, *bina … hassle*),
a spurious *nahi* ⇒ negative cue, negation scope (*cancel mat karna, bas track karo*) and keyword-driven intent
routing. v1.2 asks two separable questions:

1. Does **pretrained Hinglish knowledge** fix them? (Teacher trained on v1.1 data only.)
2. Does **more diverse, targeted data** fix them? (Teacher v2 on v1.1 + new data, same encoder and selection rule.)

and then whether a **small deployable student** (≤15M parameters, p95 < 5 ms on a laptop CPU) keeps the gains when
it inherits the teacher's soft labels while keeping v1.1's hashed word/alias/char-n-gram pieces.

Protocol is unchanged from v1.1: the exact v1.1 train/validation/test/audit splits and group ids are reused
(SHA-256 digests checked against `artifacts/results.json`); validation is the only split used for early stopping,
selection and calibration; test, audit and contrast are only scored. The v1.2 evaluator reproduces every v1.1 test
number in the whitepaper exactly (macro-F1 0.901 [0.782, 0.970] / 0.845 [0.718, 0.944], ECE, coverage, stress,
emoji pairs, audit), so all rows below are scored identically. A new 40-pair **contrast set** (Section 12.2.3) adds
per-phenomenon diagnostics.

## 12.2 Data

All additions go to TRAIN only, after removing near-duplicates (character-3-gram Jaccard > 0.6) of any validation,
test, audit or contrast message. Details and licenses: `v12/data/DATA_CARD.md`.

### 12.2.1 Public data (out of domain)

| Source | License | Raw | Kept | Use |
|---|---|---:|---:|---|
| SemEval-2020 Task 9 SentiMix, Hinglish (HF `RTT1/SentiMix`) | research-use Twitter data (re-upload tagged openrail) | 20,000 | 12,080 | labeled sentiment, intent loss masked |
| Code-mixed tweets (HF `Abhishek4896/...`) | MIT | 498 | 25 | labeled sentiment (385 exact duplicates) |
| PHINC (HF `LingoIITGN/PHINC`) | CC-BY-4.0 | 13,738 | 9,561 → 8,000 in pool | unlabeled KD pool |
| L3Cube-HingLID (GitHub `l3cube-pune/code-mixed-nlp`) | CC-BY-NC-SA-4.0 | 44,455 | 42,543 → 35,957 in pool | unlabeled KD pool |
| CMU Hinglish DoG (HF `festvox/cmu_hinglish_dog`) | CC-BY-SA-3.0 | 9,962 | 7,146 → 6,043 in pool | unlabeled KD pool |

Filters: mojibake repair, mention/URL removal, Latin script only, 3–60 words, ≥2 Romanized Hindi function words,
exact de-duplication. The unlabeled pool is capped at 50,000 by seeded sampling. **Near-duplicates of protected
messages removed: 0.** Rejected: `md-nishat-008/Code-Mixed-Sentiment-Analysis-Dataset` (machine-translated
Devanagari/Bengali/English reviews, not Romanized), L3Cube-HingCorpus (multi-GB single file; HingLID used instead).
No public Hinglish customer-support intent dataset was found.

### 12.2.2 Generated support data

1,202 messages in 591 paraphrase families (**short of the planned ~1,500**), written by the LLM assistant in three
batches across all 6 intents × 3 sentiments, seven personas (student, elderly, shopkeeper, angry repeat customer,
professional, homemaker, generic), 2–36 words (mean 11.3), 10.4% with emoji, regional spellings (*kaiku, apun,
kithe, kado*), formal Hindi and heavy shorthand. Rows are tagged for targeted phenomena: negation scope 97, neutral
with *nahi* 119, negated negative 99, varied affect words 151, praise-then-request 82, mixed intent 66.
One near-duplicate was removed. Augmentation adds 2,323 meaning-preserving variants (v1.1 shorthand/respelling
maps and character noise that never touches negators, emoji or punctuation and never produces a stress-map
spelling); 2 near-duplicates removed.

**Overlap risk.** 18 of the 32 affect words in the contrast set enter training only through the generated data,
and generated vocabulary raises coverage of validation/test vocabulary from ~0.69 to ~0.94. The generator that
wrote v1.1's test and audit sets also wrote this data.

### 12.2.3 Diagnostic contrast set

40 hand-written minimal pairs (80 messages), written before any generated data and never used for training or
selection; 10 pairs each for `negation_scope`, `neutral_nahi`, `negated_negative`, `unseen_affect`. Maximum
Jaccard to any v1.1 message 0.37. Reported as the fraction of messages with both labels correct, per phenomenon,
and the number of pairs with both members fully correct.

## 12.3 Teacher diagnostic (pretraining only)

Four encoders × three learning rates were fine-tuned on the v1.1 training split (536 rows) with two heads,
class-balanced cross-entropy, fp16 on a free Colab T4 (38 min total). Selection: lowest validation NLL.

| Encoder | Best val. NLL | Emoji handling |
|---|---:|---|
| **l3cube-pune/hing-roberta-mixed** (selected, lr 5e-5) | **0.080** | emoji are tokens |
| l3cube-pune/hing-roberta | 0.094 | emoji are tokens |
| l3cube-pune/hing-bert | 0.253 | every emoji → [UNK] |
| google/muril-base-cased | 1.143 | every emoji → [UNK] |

The BERT-vocabulary encoders cannot see emoji at all (261/908 messages contain one) and solve 0–9 of 12 emoji
contrast pairs; the RoBERTa encoders solve 12/12.

Verdict: **pretraining fixes sentiment but not intent.** Test sentiment macro-F1 1.000 vs 0.845 for v1.1
(paired +0.159 [+0.056, +0.282]); positive recall 1.00; contrast sentiment accuracy 1.00/0.95/1.00/0.90 vs
0.45/0.90/0.75/0.70. The 1.000 is a ceiling on an easy generator-style set (validation sentiment accuracy 0.943; no
test message has Jaccard > 0.6 to any training message). Intent: 0.929 vs 0.901, paired +0.032 [−0.061, +0.142],
not resolved; the teacher is *less* stable under unseen misspellings (stress consistency 92.6% vs 97.4%) and still
routes "cancel mat karna sirf order track karke batao" to `cancel_order`.

## 12.4 Teacher v2 and pseudo-labels

Same encoder and selection rule, trained on v1.1 train + generated + augmented + 4,000 public sentiment rows
(intent loss masked): 8,061 rows, 2 learning rates, 10.2 min on a T4; selected lr 5e-5, validation NLL 0.047.
The optional public-only pre-fine-tuning stage was skipped for time.

| | v1.1 | Teacher (v1.1 data) | Teacher v2 |
|---|---:|---:|---:|
| Intent macro-F1 [95% CI] | 0.901 [0.78, 0.97] | 0.929 [0.82, 1.00] | 0.966 [0.92, 0.99] |
| Paired Δ intent vs v1.1 | – | +0.032 [−0.06, +0.14] | +0.071 [−0.007, +0.172] |
| Sentiment macro-F1 | 0.845 | 1.000 | 1.000 |
| Stress intent F1 (consistency) | 0.865 (97.4%) | 0.852 (92.6%) | 0.933 (95.8%) |
| Contrast both-correct NS / NN / NG / UA | 0.60/0.65/0.40/0.55 | 0.65/0.80/0.70/0.80 | 0.80/0.85/0.70/0.95 |
| Contrast pairs fully correct | 10/40 | 24/40 | 29/40 |
| Audit errors (of 24) | 6 | 1 | 1 |

Data ablation (teacher v2 − Phase 1 teacher, identical encoder): intent +0.040 [−0.028, +0.121], P(Δ>0) = 0.85;
sentiment at ceiling for both. The negation-scope intent accuracy on the contrast set rises from 0.70 to 0.90 and
the long-standing audit error is fixed, but on the 44-group test split the data effect is not statistically
resolved. The unseen-affect improvement is not evidence of generalisation (overlap risk above). Remaining errors
are keyword-driven: "*otp nahi mila*" and complaints about delays are routed to `not_received`.

Pseudo-labels (`results/kd_stats.json`): the teacher agrees with gold on ~100% of support rows it was trained on
and 70.9% of public tweet labels; the intent-KD mask (max probability ≥ 0.6, support rows only) keeps 99.7–100% of
support rows; public and pool rows never receive intent KD.

## 12.5 Student architecture

Input pieces are exactly v1.1's hashed word (`tw:`), alias (`ta:`) and boundary-padded character 3/4-gram (`tc3:`,
`tc4:`) features, now hashed into 2^15 buckets; each token embedding is the mean of its pieces (`nn.EmbeddingBag`),
plus a learned position embedding (Eq. 4). Four pre-LayerNorm transformer blocks (d = 256, 4 heads, FFN 1024, GELU,
max 128 tokens) feed attention pooling (weights α kept for the app) and a 9-way head. As in v1.1, the logits of a
residual hashed sparse linear branch over the v1.0 feature channels are added before the heads.

| Component | Parameters |
|---|---:|
| Piece embeddings (32,768 × 256) | 8,388,608 |
| Positions (128 × 256) | 32,768 |
| 4 transformer blocks | 3,159,040 |
| Final LayerNorm + pooling + head | 3,082 |
| Hashed linear branch (32,768 × 9 + 9) | 294,921 |
| **Total** | **11,878,419** (11,583,498 without the linear branch) |

Loss: class-balanced CE on labeled rows + λ_KD = 1 · T² · KL(teacher_T ‖ student_T) with T = 2 per head (masked)
+ 0.1 · KL(p(x) ‖ p(x̃)) on meaning-preserving character noise x̃. Token dropout 0.15, AdamW (lr 5e-4,
weight decay 0.01), warm-up + cosine, early stopping on validation NLL (patience 5, ≤30 epochs). Every epoch uses all
4,061 support rows plus 4,000 re-sampled public labeled tweets and 8,000 re-sampled pool sentences. Temperature and
review threshold are fitted on validation with the v1.1 algorithm.

Deployment is torch-free: hashing, piece-embedding means and the linear branch run in NumPy; the dense encoder is
exported to ONNX (opset 17, dynamic length; torch-vs-ONNX logit parity 6.7e-7) and dynamically quantized to INT8
with onnxruntime. `python v12/predict.py "text"` returns the v1.1 JSON.

## 12.6 Results

**[PENDING 03]** Full table from `results/phase5_table.md`: v1.1 | teacher (v1.1 data) | teacher v2 | student + KD |
student no-KD | student w/o linear | seeds 43/44 | deployed INT8 — intent/sentiment macro-F1 with CI and paired Δ
vs v1.1, positive recall, stress F1 and consistency, emoji pairs, audit errors, contrast per phenomenon, ECE,
coverage/accepted accuracy, parameters, latency.

**[PENDING 03]** KD effect (student + KD vs no-KD on identical labeled data), linear-branch effect, seed spread,
INT8 accuracy delta.

### Latency (Mac CPU; architecture measured, **[PENDING 03]** re-measured with the trained weights)

Apple M2, Python 3.12, onnxruntime 1.30; 80 warm-up + 600 timed calls, batch 1, including hashing, embedding mean,
encoder, linear branch, calibration and JSON serialization (architecture-only measurement with untrained weights,
`runs/smoke/latency_smoke.json`):

| Configuration | Test messages p50/p95/p99 (ms) | 512 code points p95 (ms) |
|---|---|---:|
| INT8, 1 thread | 0.65 / 0.92 / 1.04 | 6.39 |
| INT8, 4 threads | 0.54 / 0.86 / 1.10 | 3.84 |
| FP32, 1 thread | 1.18 / 1.70 / 1.83 | 11.46 |
| FP32, 4 threads | 0.72 / 1.25 / 2.03 | 4.69 |
| v1.1 (same machine) | – / 0.55 / – | 2.02 |

Typical messages are far inside the 5 ms budget; the 512-code-point worst case meets it only with 4 intra-op
threads.

## 12.7 What improved and what did not

**[PENDING 03]** for the student. Final for the teachers:

- Improved: sentiment (all sets), positive recall, emoji contrast pairs, the *nahi* ⇒ negative cue, audit errors,
  negation-scope intent on the contrast set (teacher v2), robustness to unseen misspellings (teacher v2).
- Not resolved: intent gains over v1.1 and over the Phase 1 teacher on the 44-group development test (CIs include 0);
  keyword-driven routing to `not_received`; feedback about logistics routed to operational intents.

## 12.8 Threats to validity

- **Same generator.** The v1.1 test and audit sets, most v1.1 training data, the 1,202 generated messages and the
  contrast set were all written by LLMs (the latter two by the same assistant). Gains may partly reflect matching the
  generator's style and vocabulary (validation/test vocabulary coverage rises from ~0.69 to ~0.94 with generated
  data) rather than real-world robustness. The 1.000 sentiment scores are a ceiling on this distribution.
- **Contrast leakage of vocabulary.** 18 contrast affect words appear in training only via generated data;
  `unseen_affect` results after Phase 2 are not generalisation evidence.
- **Out-of-domain public data.** SentiMix/PHINC/HingLID/CMU DoG are political, cricket, movie and chit-chat text with
  visible label noise (teacher–gold agreement 70.9%), not customer support; licenses restrict redistribution.
- **Pseudo-label noise.** Soft labels on 50,000 unlabeled tweets come from a teacher that never saw that domain with
  labels for intent (intent KD masked there) and whose sentiment agreement with noisy gold is 71%.
- **Seeds.** Teachers use one seed; student seeds 43/44 run only if GPU time allowed **[PENDING 03]**.
- **Statistical power.** 44 test groups; most intent differences between strong systems are not resolved.
- **Free-GPU training.** Colab T4 sessions; optimizer moments are not checkpointed for teachers (weights, schedule
  and RNG are); no hyper-parameter search beyond ≤3 learning rates; Stage 1 public pre-fine-tuning skipped.
- **Human data.** No real customer messages were available (`data/human_test.jsonl` hook unused).

## 12.9 Requirements scorecard (v1.2)

| Requirement | Status | Evidence |
|---|---|---|
| ≤15M parameters (student) | Met | 11,878,419 (Section 12.5) |
| p95 < 5 ms at batch 1 on the Mac CPU, incl. hashing | Met for typical messages; worst case needs 4 threads | 0.92 ms test messages; 512 code points 6.39 ms (1 thread) / 3.84 ms (4 threads) **[re-measure 03]** |
| Exact v1.1 splits; no test/audit use for choices | Met | digest-checked export; selection/calibration on validation only |
| Near-duplicate filtering of new data | Met | removed: public 0, generated 1, augmented 2 |
| Hash parity Colab ↔ Mac | Met | 50/50 fixtures on Python 3.10/3.12/3.13/3.14 |
| Sentiment robustness | Improved (teachers) | Section 12.3–12.4 **[student PENDING 03]** |
| Negation scope | Partially improved (teacher v2) | contrast scope intent 0.90; 6/10 scope pairs |
| Real-world validation | Not met | no human-collected data |
