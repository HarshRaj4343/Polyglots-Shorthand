# Section 12 (draft): Version 1.2 — Teacher Distillation into a Hashed-Piece Transformer

> Status: DRAFT, all phases complete. Every number comes from `v12/results/*.json`.

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

Development test (190 rows, 44 groups); macro-F1 with 95% group-bootstrap interval; Δ is paired against v1.1 on the
same 5,000 resamples. Full table with ECE and coverage: `results/phase5_table.md`.

| System | Params | Intent F1 [CI] | Δ intent | Sentiment F1 [CI] | Δ sentiment | Pos. recall | Stress F1 int/sent (consistency) | Emoji | Audit err. | Contrast pairs |
|---|---:|---|---|---|---|---:|---|---:|---:|---:|
| v1.1 | 0.83M | 0.901 [0.78, 0.97] | – | 0.845 [0.72, 0.94] | – | 0.72 | 0.865/0.826 (97.4%/92.6%) | 9/12 | 6 | 10/40 |
| Teacher (v1.1 data) | 278M | 0.929 [0.82, 1.00] | +0.032 [−0.06, +0.14] | 1.000 | +0.159 [+0.06, +0.28] | 1.00 | 0.852/1.000 (92.6%/100%) | 12/12 | 1 | 24/40 |
| Teacher v2 | 278M | 0.966 [0.92, 0.99] | +0.071 [−0.01, +0.17] | 1.000 | +0.159 [+0.06, +0.28] | 1.00 | 0.933/1.000 (95.8%/100%) | 12/12 | 1 | 29/40 |
| Student + KD (s42) | 11.88M | 0.914 [0.81, 0.98] | +0.014 [−0.10, +0.13] | 0.911 [0.82, 0.98] | +0.068 [−0.07, +0.21] | 1.00 | 0.914/0.917 (97.9%/96.3%) | 12/12 | 2 | 17/40 |
| Student no-KD (s42) | 11.88M | 0.888 [0.77, 0.97] | −0.011 [−0.12, +0.11] | 0.878 [0.77, 0.96] | +0.034 [−0.07, +0.14] | 0.93 | 0.895/0.836 (96.8%/92.6%) | 12/12 | 3 | 18/40 |
| Student + KD w/o linear (s42) | 11.58M | 0.958 [0.89, 0.99] | +0.062 [−0.03, +0.17] | 0.909 [0.81, 0.98] | +0.066 [−0.04, +0.17] | 0.93 | 0.918/0.925 (94.7%/96.3%) | 11/12 | 1 | 20/40 |
| Student + KD (s43) | 11.88M | 0.911 [0.82, 0.98] | +0.014 [−0.09, +0.14] | 0.853 [0.74, 0.94] | +0.009 [−0.11, +0.13] | 0.95 | 0.910/0.813 (98.9%/88.9%) | 12/12 | 3 | 17/40 |
| Student + KD (s44) | 11.88M | 0.919 [0.82, 0.98] | +0.020 [−0.09, +0.13] | 0.952 [0.87, 1.00] | +0.110 [−0.00, +0.23] | 0.93 | 0.914/0.925 (97.9%/97.4%) | 12/12 | 1 | 17/40 |
| **Student + KD, ONNX INT8 (deployed)** | 11.88M | 0.914 [0.81, 0.98] | +0.014 [−0.10, +0.13] | 0.916 [0.83, 0.98] | +0.073 [−0.06, +0.22] | 1.00 | 0.914/0.917 (97.9%/95.8%) | 12/12 | 2 | 17/40 |

### Paired comparisons (same resamples)

| Comparison | Intent Δ [95% CI], P(Δ>0) | Sentiment Δ [95% CI], P(Δ>0) |
|---|---|---|
| KD effect: student + KD − no-KD (s42) | +0.025 [−0.034, +0.113], 0.70 | +0.034 [−0.055, +0.124], 0.78 |
| Linear branch: with − without (s42) | −0.048 [−0.124, +0.014], 0.07 | +0.002 [−0.091, +0.094], 0.51 |
| Student + KD − teacher v2 | −0.057 [−0.136, +0.004], 0.04 | **−0.092 [−0.184, −0.020]**, 0.00 |
| INT8 − FP32 (s42) | +0.000 [+0.000, +0.000] | +0.005 [+0.000, +0.017] |
| Data ablation: teacher v2 − teacher (v1.1 data) | +0.040 [−0.028, +0.121], 0.85 | 0.000 (both at ceiling) |

Seed spread (student + KD, seeds 42/43/44): intent mean 0.915, sd 0.004; **sentiment mean 0.906, sd 0.049**.
Best validation NLL varied from 0.065 (s42) to 0.156 (s43).

### Contrast set (both labels correct; pairs fully correct)

| System | Negation scope | Neutral *nahi* | Negated negative | Unseen affect† | Pairs |
|---|---|---|---|---|---:|
| v1.1 | 0.60 (3) | 0.65 (4) | 0.40 (1) | 0.55 (2) | 10/40 |
| Teacher v2 | 0.80 (6) | 0.85 (8) | 0.70 (6) | 0.95 (9) | 29/40 |
| Student + KD (s42) | 0.65 (5) | 0.75 (6) | 0.45 (1) | 0.70 (5) | 17/40 |
| Student no-KD (s42) | 0.60 (4) | 0.75 (7) | 0.45 (1) | 0.75 (6) | 18/40 |
| Student + KD w/o linear | 0.70 (5) | 0.80 (7) | 0.40 (1) | 0.80 (7) | 20/40 |

† affect words present in generated training data (not a generalisation test).

### Calibration

Deployed INT8 student: temperatures fitted on validation T_I = 0.40 (grid boundary), T_S = 0.755; review thresholds
0.0 for both heads (every validation prediction qualified), so coverage is 1.00; ECE 0.070 (intent) / 0.045
(sentiment) vs 0.082 / 0.106 for v1.1. The review flag therefore never fires for the student on this split — unlike
v1.1, whose sentiment head accepted 75.3%.

### Quantization and export

ONNX FP32 parity with PyTorch: 5.8e-6 maximum absolute logit difference. INT8 dynamic quantization: identical intent
predictions on every validation/test/stress/audit/contrast message, one sentiment change on 190 test messages;
encoder file 12.8 MB → 3.4 MB (embedding and linear tables stay float32 in a 34.7 MB NumPy archive).

### Latency (trained model, Mac CPU)

Apple M2, Python 3.12, onnxruntime 1.30; 80 warm-up + 600 timed calls, batch 1, including hashing, embedding mean,
encoder, linear branch, calibration and JSON serialization (`results/latency_student_kd_s42.json`).

| Configuration | Test p50 / p95 / p99 (ms) | p95 at 32 / 128 / 512 code points (ms) | msg/s |
|---|---|---|---:|
| INT8, 1 thread | 0.64 / 0.92 / 1.15 | 0.55 / 1.68 / 5.98 | 1,497 |
| **INT8, 2 threads (default)** | **0.57 / 0.81 / 0.90** | 0.49 / 1.31 / **4.29** | 1,686 |
| INT8, 4 threads | 0.53 / 0.77 / 0.95 | 0.49 / 1.30 / 3.45 | 1,782 |
| FP32, 1 thread | 1.14 / 1.62 / 1.75 | 0.90 / 2.98 / 10.96 | 867 |
| FP32, 2 threads | 0.81 / 1.14 / 1.23 | 0.68 / 1.97 / 6.83 | 1,201 |
| FP32, 4 threads | 0.68 / 1.00 / 1.39 | 0.61 / 1.65 / 4.91 | 1,388 |
| v1.1 (same machine) | 0.27 / 0.38 / 0.45 | 0.25 / 0.55 / 1.40 | – |

At 512 code points (106 tokens) the ONNX encoder dominates (≈4.3 of 6.0 ms at one thread; hashing ≈1.1 ms).

## 12.7 What improved and what did not

Improved (with evidence strength):
- **Teachers, sentiment**: resolved gains on every set (paired +0.159 [+0.056, +0.282]).
- **Teacher v2, negation scope and robustness**: contrast negation-scope intent 0.90 vs 0.65; stress intent 0.933;
  audit 1 error vs 6; 29/40 contrast pairs vs 10. Intent vs v1.1 +0.071 [−0.007, +0.172]: borderline.
- **Deployable student**: 11.88M parameters, p95 0.81 ms (4.29 ms worst case), INT8 lossless on these sets;
  unseen-misspelling stability at least as good as v1.1 (consistency 97.9% vs 97.4%); emoji pairs 12/12;
  positive recall 1.00; contrast pairs 17/40 vs 10/40.

Not improved or not resolved:
- **The student does not keep the teacher's gains.** Sentiment is significantly below teacher v2 (−0.092
  [−0.184, −0.020]); its gains over v1.1 are not resolved for either head, and sentiment varies strongly across seeds
  (0.853–0.952).
- **KD is not shown to help**: +0.025 / +0.034 over the no-KD student, CIs include 0; the no-KD student even solves
  one more contrast pair.
- **The hashed linear branch appears to hurt intent** (−0.048 [−0.124, +0.014], P(Δ>0) = 0.07; without it intent is
  0.958 and 20/40 contrast pairs). The deployed model keeps it because it was the pre-registered primary variant; the
  no-linear variant was selected by nothing and would need confirmation on more seeds before switching.
- **Negated negatives** stay at 1/10 pairs for every student; students also regress on negation scope relative to
  teacher v2 (5/10 vs 6/10 pairs), and seed 43 reintroduces the "cancel mat karna" audit error.
- **Calibration**: the review threshold collapses to 0 for the student, so it never defers to a human on this split.

## 12.8 Threats to validity

- **Same generator.** The v1.1 test and audit sets, most v1.1 training data, the 1,202 generated messages and the
  contrast set were all written by LLMs (the latter two by the same assistant). Gains may partly reflect matching the
  generator's style and vocabulary (validation/test vocabulary coverage rises from ~0.69 to ~0.94 with generated
  data) rather than real-world robustness. The 1.000 sentiment scores are a ceiling on this distribution.
- **Contrast leakage of vocabulary.** 18 contrast affect words appear in training only via generated data;
  `unseen_affect` results after Phase 2 are not generalisation evidence.
- **Out-of-domain public data.** SentiMix/PHINC/HingLID/CMU DoG are political, cricket, movie and chit-chat text with
  visible label noise (teacher–gold agreement 70.9%), not customer support; licenses restrict redistribution.
- **Pseudo-label noise.** Soft labels on 50,000 unlabeled tweets come from a teacher trained on 4,000 of them plus
  support data; on support rows the teacher reproduces the gold labels it was trained on.
- **Seeds.** Teachers use one seed. Students use three seeds only for the primary variant; the KD and linear-branch
  ablations use seed 42 only, while the primary variant's sentiment sd across seeds is 0.049 — larger than both
  ablation effects.
- **Statistical power.** 44 test groups; most differences between strong systems are not resolved.
- **Free-GPU training.** Colab T4 sessions; optimizer moments are not checkpointed for teachers (weights, schedule
  and RNG are); ≤3 learning rates; Stage 1 public pre-fine-tuning skipped; no student hyper-parameter search.
- **Human data.** No real customer messages were available (`data/human_test.jsonl` hook unused).

## 12.9 Requirements scorecard (v1.2)

| Requirement | Status | Evidence |
|---|---|---|
| ≤15M parameters (student) | Met | 11,878,419 |
| p95 < 5 ms at batch 1 on the Mac CPU, incl. hashing | Met (2 threads) | test messages 0.81 ms; 512 code points 4.29 ms (5.98 ms with 1 thread); p99 at 512 code points not measured below 5 ms |
| Exact v1.1 splits; no test/audit use for choices | Met | digest-checked export; selection/calibration on validation only |
| Near-duplicate filtering of new data | Met | removed: public 0, generated 1, augmented 2 |
| Hash parity Colab ↔ Mac | Met | 50/50 fixtures on Python 3.10/3.12/3.13/3.14 in every notebook |
| Sentiment robustness | Teachers: met; student: not resolved | teacher +0.159 [+0.056, +0.282]; student +0.068 [−0.07, +0.21], seed sd 0.049 |
| Negation scope | Partial | teacher v2 6/10 scope pairs; deployed student 5/10 |
| Distillation keeps teacher gains | Not met | student − teacher v2 sentiment −0.092 [−0.184, −0.020] |
| Real-world validation | Not met | no human-collected data |
