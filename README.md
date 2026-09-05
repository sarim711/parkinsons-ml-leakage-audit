## Parkinson's Severity Estimation under Patient-Disjoint Evaluation

## Overview

This project estimates **Parkinson's disease severity** from voice recordings in the UCI
Parkinson's Telemonitoring dataset, and shows that commonly reported results on this benchmark
are inflated by **data leakage**.

Each patient in the dataset contributes roughly **140 recordings**, and their severity score
barely changes across those recordings. When cross-validation splits on **rows** rather than on
**patients**, recordings from the same person land in both the training and test sets. A model
can then recognise the patient instead of reading the voice signal, and the reported score
mostly measures patient recall.

The same pipeline is evaluated under both protocols. The only thing that changes between them
is the splitter. The gap between the two is the size of the leakage artefact.

**Project history**: this repository began as an Explainable AI and adversarial-robustness
study, perturbing voice features to measure the effect on model performance. Building a
baseline to perturb against surfaced the leakage problem first, so the question changed to what
the benchmark actually supports once patients are held out properly.

---

## Dataset

- **Dataset Used**: `data/raw/parkinsons_updrs.data`

- **Data Source**:

  - UCI Parkinson's Telemonitoring, **5,875 sustained-vowel recordings** from **42 early-stage
    patients** recorded over six months.
  - Available from the
    [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring).

- **Target Variable**:

  - `motor_UPDRS`, the continuous motor score, predicted directly by the regressors.

- **Severity Labels** (derived from `motor_UPDRS`):

  - Binary: Mild `< 20` vs Not-Mild `>= 20`
  - 3-class: Mild `< 20`, Moderate `[20, 30)`, Severe `>= 30`

- **Grouping Variable**:

  - `subject_id`, used to keep all of a patient's recordings inside a single fold.

- **Attribution**:

  - A. Tsanas, M. A. Little, P. E. McSharry, L. O. Ramig (2010). *Accurate telemonitoring of
    Parkinson's disease progression by non-invasive speech tests.* IEEE Transactions on
    Biomedical Engineering.

---

## Methodology

### Features

- **39 features**: 3 demographics (`age`, `sex`, `test_time`), 16 raw voice measures
  (jitter, shimmer, NHR, HNR, RPDE, DFA, PPE), 11 log-transformed jitter and shimmer variants,
  4 ratios and 5 age interactions.
- A separate **80-column patient-aggregate block** (per-patient mean, std, min, max, median of
  the 16 raw measures) is used only for the aggregation experiment.

### Evaluation

- `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, grouped by `subject_id`,
  so **no patient appears in both training and test**. This is asserted at run time in
  `evaluate.py` and independently in `tests/`.
- All transforms are **fitted on the training fold only**. `PowerTransformer` defaults to
  `standardize=True`, so Ridge and SVR are standardised implicitly by the Yeo-Johnson step.
- **Regression then threshold**: regressors predict continuous `motor_UPDRS`, thresholded at
  **20** to recover the binary label.
- Models span linear (Logistic Regression, Ridge, SVR), tree ensembles (Random Forest,
  Extra Trees), gradient boosting (XGBoost, LightGBM, Gradient Boosting) and neural networks
  (MLP, deeper MLP, FT-Transformer-lite).
- Metrics are mean and sample standard deviation (ddof=1) across the 5 folds. Seed fixed at
  **42**, hyperparameters fixed as literals in `models.py`, so results reproduce on every run.

---

## Key Findings & Results

### The Leakage Result

Identical features, model (Extra Trees regressor), transform (Yeo-Johnson), threshold (20) and
seed. Only the splitter differs.

- **Row-level `StratifiedKFold`**: MAE `0.61`, AUC `0.998`, Accuracy `0.982`
- **Patient-disjoint `StratifiedGroupKFold`**: MAE `7.38`, AUC `0.629`, Accuracy `0.562`
- **Difference**: MAE `6.77`, AUC `0.369`, Accuracy `0.421`

Under the row-level protocol, **all 42 patients appear in both training and test in every
fold**. Under the patient-disjoint protocol the count is zero. The row-level figures fall in
the range commonly reported for this benchmark.

### Model Performance

- Best pipeline under honest evaluation: **Extra Trees regressor with Yeo-Johnson**, thresholded
  at 20, giving MAE `7.38` and AUC `0.629`.
- It does not beat plain Extra Trees on AUC (`0.629` against `0.633`), and per-fold AUC ranges
  from `0.798` down to `0.370`, so one held-out patient group is predicted worse than chance.
- Direct binary classification is weaker than regression then threshold, topping out near
  AUC `0.55`.
- 3-class macro-F1 sits between `0.29` and `0.37` for every model family. The severe class,
  present in only 6 patients, is the bottleneck.
- Per-model numbers for every track are written to `results/*.csv` by `run.py`.

### Patient-Aggregate Features: a Negative Result

- Appending the 80-column patient-aggregate block (119 features total) **hurts** performance,
  giving AUC `0.459` to `0.489` against `0.553` for the 39-feature baseline.
- The aggregates act as a per-patient fingerprint that the trees memorise and cannot transfer
  to unseen patients. This is a common route by which leakage re-enters this benchmark.

### Hyperparameter Provenance

- The search that produced the values in `models.py` **was not recorded**, and has not been
  reconstructed after the fact. `python tune.py provenance` states what is and is not known.
- It makes little difference: a 16-point sensitivity sweep moves MAE by `0.139` and AUC by
  `0.020` across the whole grid, against `6.77` and `0.369` for the splitter choice, and the
  shipped configuration ranks 7th of 16 on both metrics. Nested cross-validation gives MAE
  `7.376` against `7.384` non-nested, so selection optimism is negligible.

### Conclusion

Once patients are held out properly, the voice features in this dataset support only weak
severity prediction for an unseen patient, around AUC `0.63`. The near-perfect scores reported
elsewhere are a property of the evaluation protocol, not of the models. Any result on this
benchmark that does not state its splitter should be read as the row-level figure.

---

## Repository Structure

- `preprocessing.py`
  - Data loading, the 39-feature block, patient-aggregate features, and patient-disjoint folds.

- `models.py`
  - Model factory functions for every classifier and regressor, with fixed hyperparameters.

- `evaluate.py`
  - Cross-validated evaluation for classification and regression-then-threshold, with per-fold
    preprocessing and the fold-disjointness guard.

- `run.py`
  - Orchestrator. Runs a section, prints metrics, and writes CSV or JSON to `results/`.

- `tune.py`
  - Hyperparameter provenance, the sensitivity sweep, and nested cross-validation. Not on the
    default path and not required to reproduce any reported number.

- `neural.py`
  - PyTorch models: MLP, deeper MLP, and an FT-Transformer-lite feature-attention model.
    Optional, requires `torch`.

- `parkinsons_leakage_analysis.ipynb`
  - Single self-contained notebook covering the full analysis, including the patient-mean
    oracle, a diversity ablation, the aggregation negative result, and few-shot
    personalisation. Committed with outputs.

- `tests/test_folds.py`
  - Asserts that folds are patient-disjoint and that transforms are fitted on training rows
    only. Run in CI on every push.

- `data/raw/`
  - The dataset and its UCI variable descriptions.

- `results/`
  - Generated by `run.py` and `tune.py`. Not committed.

---

## How to Run

1. Clone this repository to your local machine.

2. Install the pinned dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Reproduce the central result:

   ```bash
   python run.py leakage-demo
   ```

4. Run any other section. Valid sections are `regression-final`, `regression`, `2class`,
   `3class`, `literature`, `leakage-demo` and `all`:

   ```bash
   python run.py regression-final
   ```

5. Inspect hyperparameter provenance and selection optimism:

   ```bash
   python tune.py provenance
   python tune.py sensitivity
   python tune.py nested
   ```

6. Run the tests:

   ```bash
   pip install pytest
   python -m pytest tests/ -v
   ```

Results are written to `results/`, one CSV per section. The tree ensembles use 500 to 700 trees
and are CPU-heavy, so run sections individually rather than `all` if you are core-limited.

Reported numbers were produced with Python 3.14.6 and the exact versions pinned in
`requirements.txt`.
