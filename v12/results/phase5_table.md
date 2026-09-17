| System | Params | Intent F1 [95% CI] | Δ vs v1.1 | Sentiment F1 [95% CI] | Δ vs v1.1 | Pos. recall | Stress F1 int/sent (consist.) | Emoji pairs | Audit err. | Contrast both-correct NS/NN/NG/UA (pairs) | ECE int/sent | Coverage / acc. accepted (sent) | p50/p95/p99 ms |
|---|---:|---|---|---|---|---:|---|---:|---:|---|---|---|---|
| v1.1 full (NumPy linear + attention) | 827,721 | 0.901 [0.78, 0.97] | - | 0.845 [0.72, 0.94] | - | 0.72 | 0.865/0.826 (97.4%/92.6%) | 9/12 | 6 | 0.60/0.65/0.40/0.55 (10/40) | 0.082/0.106 | 0.75 / 0.951 | not measured |
| Teacher, v1.1 data (Phase 1) | 278,050,569 | 0.929 [0.82, 1.00] | +0.032 [-0.06, +0.14] | 1.000 [1.00, 1.00] | +0.159 [+0.06, +0.28] | 1.00 | 0.852/1.000 (92.6%/100.0%) | 12/12 | 1 | 0.65/0.80/0.70/0.80 (24/40) | 0.070/0.014 | 1.00 / 1.000 | not measured |
| Teacher v2 (Phase 3) | 278,050,569 | 0.966 [0.92, 0.99] | +0.071 [-0.01, +0.17] | 1.000 [1.00, 1.00] | +0.159 [+0.06, +0.28] | 1.00 | 0.933/1.000 (95.8%/100.0%) | 12/12 | 1 | 0.80/0.85/0.70/0.95 (29/40) | 0.044/0.000 | 1.00 / 1.000 | not measured |
| Student + KD (seed 42) | not run | | | | | | | | | | | | |
| Student no-KD (seed 42) | not run | | | | | | | | | | | | |
| Student + KD w/o linear branch | not run | | | | | | | | | | | | |
| Student + KD (seed 43) | not run | | | | | | | | | | | | |
| Student + KD (seed 44) | not run | | | | | | | | | | | | |
| Student + KD, deployed ONNX INT8 | not run | | | | | | | | | | | | |

Data ablation (teacher v2 − Phase 1 teacher, same encoder): intent +0.040 [-0.028, +0.121], P(Δ>0)=0.85; sentiment +0.000 [+0.000, +0.000], P(Δ>0)=0.00
