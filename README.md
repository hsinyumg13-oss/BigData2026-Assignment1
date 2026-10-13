# BigData2026 Assignment 1 — Analysis of SVD for Fine-Tuning LLMs

NYCU Big Data Analytics Techniques and Applications, Fall 2026

This repository fine-tunes `meta-llama/Llama-3.2-1B` on MetaMathQA with PiSSA-style SVD initialization and evaluates it on GSM8K and MATH (1,000 test samples each). The code is modified from the course-provided notebooks, which are adapted from [PiSSA](https://github.com/MuLabPKU/PiSSA).

## Files

| File | Description |
|---|---|
| `finetune_q1-q3.ipynb` | Fine-tuning and evaluation for Q1–Q3 (output of the Q3 run) |
| `finetune_q4.ipynb` | SVD analysis of `q_proj` in layers 0, 7, 15 for Q4 |
| `q3_bar_chart.ipynb` | Bar chart of Q3 accuracy |
| `v1.csv`, `v2.csv`, `v3.csv` | `benchmark_results.csv` of the three Kaggle runs |

## Modifications to the provided code

**`finetune_q1-q3.ipynb`**
- **Q2:** `train_one_task()` records wall-clock time between the tagged start and end of training and prints `Training time: X minutes`.
- **Q3:** `pissa_from_linear_exact()` supports three additional singular-value ranges (`n = len(S)`):
  - `p25`: `start = n // 4`
  - `p50`: `start = n // 2`
  - `bottom`: `start = n - rank`

**`finetune_q4.ipynb`**
- Computes SVD, cumulative energy, the minimum rank for >= 50% energy, and the 6 required charts.
- The last cell computes the energy share of the four rank-16 ranges, which is used to explain Q3(b).

## Run settings (Kaggle, GPU T4 x2)

| Run | `RUN_FRESH_BASELINE` | `MANUAL_RUNS` | Result file |
|---|---|---|---|
| V1 | True | rank 16, `default` | `v1.csv` |
| V2 | False | rank 64 and 256, `default` | `v2.csv` |
| V3 | False | rank 16, `p25` / `p50` / `bottom` | `v3.csv` |

## Results

**Q1 & Q2**

| | Original | Rank 16 | Rank 64 | Rank 256 |
|---|---|---|---|---|
| GSM8K accuracy | 3.5% | 15.3% | 20.4% | 27.1% |
| MATH accuracy | 0.1% | 3.8% | 4.8% | 5.3% |
| Training time (min) | – | 64.21 | 68.30 | 92.42 |

**Q3 (rank 16)**

| | Top | 25% | 50% | Bottom |
|---|---|---|---|---|
| GSM8K accuracy | 15.3% | 12.6% | 11.7% | 11.4% |
| MATH accuracy | 3.8% | 3.1% | 2.1% | 2.6% |

**Q4: minimum rank for >= 50% cumulative energy (`q_proj`)**

| Layer 0 | Layer 7 | Layer 15 |
|---|---|---|
| 62 | 120 | 84 |

## Reference

- Meng, F., Wang, Z., & Zhang, M. (2024). PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models. NeurIPS 2024.
- Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
