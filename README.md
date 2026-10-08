# Forecasting Event Counts in Topological Networks: An Empirical Comparison of Models with Learned Connectivity Weights

**Proceedings:** Proceedings of the Autumn Conference of the Korean Data & Information Science Society, 2026, 1–1.

**Authors:** 김성택¹ · 조민영²

¹ 단국대학교 정보통계학과 · ² 단국대학교 통계데이터사이언스학과

## Abstract

Forecasting event counts in networks with known topology requires accounting for both physical connectivity and dependencies among event types. Recent models, including TNPAR (Topological Neural Poisson Auto-Regressive Model) and S²GCSL (Simple yet Scalable Granger Causal Structural Learning), have advanced structural learning in this setting, but their emphasis on structural recovery leaves open whether their connectivity representations are well suited to count forecasting. We investigate whether learned continuous connectivity weights offer a more effective representation for prediction through an empirical comparison of TNPAR, a direct variational model, a continuous connectivity weight model, S²GCSL, and THP (Topological Hawkes Process). Experiments use the public 24V 439N Microwave dataset, containing 439 devices and 24 alarm types. Event occurrences are aggregated into 60-second intervals and divided chronologically into training, validation, and test sets. The continuous connectivity weight model directly applies learned weights to historical inputs to produce Poisson predictive means. Across three training repetitions on the same dataset, it achieves the lowest mean predictive negative log-likelihood among the five models, at 0.001517, representing reductions of 17.3% relative to TNPAR and 16.9% relative to the direct variational model. It also achieves the lowest MAE and RMSE among the five models, with its leading prediction rank maintained across all three repetitions. Its average precision against the provided reference connectivity labels is 0.300, exceeding TNPAR (0.278), the direct variational model (0.293), and THP (0.244), but remaining below S²GCSL (0.335). These findings demonstrate predictive benefits of learned continuous connectivity weights in this setting and motivate extensions to additional network datasets and connectivity structures that vary over time.

**Keywords:** Event Count Forecasting; Topological Event Sequences; Learned Connectivity Weights; Poisson Models; Granger Causality.

**Acknowledgment:** 본 연구는 2026년 과학기술정보통신부 및 정보통신기획평가원의 SW중심대학사업 지원을 받아 수행되었음 (2024-0-00035).

## Main real-data results

The table reports the 24V_439N_Microwave test-set results. For TNPAR, direct variational, deterministic-connectivity, and S²GCSL models, values are mean ± sample standard deviation over three training runs on the same chronological split. THP was run once; its standard deviation is therefore unavailable. Lower NLL, MAE, and RMSE are better; higher AP is better.

| Model | Predictive NLL | MAE | RMSE | AP | F1 |
|---|---:|---:|---:|---:|---:|
| TNPAR-style encoder | 0.001835 ± 0.000004 | 0.000860 ± 0.000009 | 0.021882 ± 0.000031 | 0.278 ± 0.007 | 0.000 |
| Direct variational | 0.001826 ± 0.000014 | 0.000896 ± 0.000010 | 0.021614 ± 0.000064 | 0.293 ± 0.007 | 0.000 |
| Deterministic connectivity weights | **0.001517 ± 0.000007** | **0.000615 ± 0.000019** | **0.019626 ± 0.000066** | 0.300 ± 0.002 | 0.000 |
| S²GCSL-style fixed-window | 0.002018 ± 0.000008 | 0.000973 ± 0.000016 | 0.024362 ± 0.000195 | **0.335 ± 0.005** | 0.074 ± 0.031 |
| THP-style EM/BIC | 0.002000 (1 run) | 0.000897 (1 run) | 0.024495 (1 run) | 0.244 (1 run) | 0.110 (1 run) |

These are empirical results for one public network dataset and one chronological split, not evidence of general superiority. The three neural-model repetitions change training randomness, not the underlying data split. THP has one run. The implementations are paper-informed adaptations rather than exact reproductions of all authors' original code. AP uses the dataset's provided reference graph; it is not verified ground-truth Granger causality. Neural-model F1 uses a fixed 0.5 threshold; no threshold was tuned on the test set.

## Repository contents

- `src/comparison_v2/`: model, training, data-loading, protocol, and CPU test code for the five-model study.
- `src/data.py`, `src/evaluation.py`: simulation utilities and evaluation metrics.
- `results/`: verified result interpretation and the three-run repetition audit.
- `data/prepared/simulation_*`: compact prepared simulation inputs used by the study; these reuse previously examined simulation paths and are not new independent repetitions.
- `data/source_manifest.json`: pinned public-source URLs, revisions, byte counts, and SHA-256 hashes for the real dataset and cited gCastle implementation.
- `scripts/fetch_public_data.py`, `scripts/prepare_real_data.py`: download/hash verification and real-data preparation. Raw real-data files are not committed.

## Recreate the real-data input

Python 3.10+ is recommended. Install dependencies, download the pinned source archive, and prepare the count grid:

```powershell
python -m pip install -r requirements.txt
python scripts/fetch_public_data.py
python scripts/prepare_real_data.py
```

The downloader checks the pinned revision, exact byte lengths, and SHA-256 digests before accepting files. It does not read credentials. The raw download is written under the ignored `data/raw/` directory. The prepared real input is written to `data/prepared/real_24V_439N_Microwave/`.

To run the full five-model study, use the saved protocol and a suitable accelerator (the original completed runs used Kaggle T4 GPUs):

```powershell
python src/comparison_v2/study_v2.py --data-root data/prepared --output outputs/real_run --datasets real_24V_439N_Microwave --device cuda:0
```

The full study is computationally intensive. Use `--device cpu` only for small tests, not as a runtime-equivalent substitute for the reported GPU runs. Existing result files are retained as verification records; reproducing them requires rerunning training and validation-based model selection.

## Data and method notes

The real dataset is the public `24V_439N_Microwave` alarm dataset from the gcastle-hub dataset repository, pinned to commit `1858e9c39d90a524e7d8ff8b9456f4b1f231ff03`. Alarm occurrences are binned into 60-second intervals. The chronological split is TRAIN 5–5183, VALIDATION 5183–6911, and TEST 6911–8639. Physical topology is known input; the provided DAG is used only as a reference label for graph-score evaluation.

TNPAR, S²GCSL, and THP entries are explicitly labeled adaptations. All models receive common binned occurrence histories and physical topology; event-end durations are not used as model inputs. The study compares forecasts of observed counts, not latent true intensity λ. See `src/comparison_v2/protocol.json` for model-specific settings and deviations from the source papers.

## References

- Liu, Y. et al. (2024). *TNPAR: Topological Neural Poisson Auto-Regressive Model for Learning Granger Causal Structure from Event Sequences*. AAAI 2024. <https://doi.org/10.1609/aaai.v38i18.30033>
- Li, M. et al. (2024). *A Simple yet Scalable Granger Causal Structural Learning Approach for Topological Event Sequences*. NeurIPS 2024.
- Cai, R. et al. (2021). *THP: Topological Hawkes Processes for Learning Causal Structure on Event Sequences*. <https://arxiv.org/abs/2105.10884>
- gcastle-hub. *Dataset: alarm / 24V_439N_Microwave*, pinned revision: <https://github.com/gcastle-hub/dataset/tree/1858e9c39d90a524e7d8ff8b9456f4b1f231ff03/alarm>
- Huawei Noah's Ark Lab. gCastle TTPM implementation, pinned revision: <https://github.com/huawei-noah/trustworthyAI/blob/8675af0bc532c7f138158ded62e8cf303e9af655/gcastle/castle/algorithms/ttpm/ttpm.py>
