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
    [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)
    (doi:10.24432/C5ZS3N) under a CC BY 4.0 license. The MIT license of this repository covers the
    code only.

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
  `standardize=True`, so Ridge, Huber and SVR are standardised implicitly by the Yeo-Johnson step.
- **Regression then threshold**: regressors predict continuous `motor_UPDRS`, thresholded at
  **20** to recover the binary label.
- Models span linear (Logistic Regression, Ridge, Huber, SVR), tree ensembles (Random Forest,
  Extra Trees), gradient boosting (XGBoost, LightGBM, Gradient Boosting) and neural networks
  (MLP, deeper MLP, FT-Transformer-lite).
- Two baselines use no voice features: the **overall average** (the training-fold mean for
  every recording) and the **patient average** (the mean of the same patient's other
  recordings, i.e. what knowing the patient alone gives).
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

The row-level score does not need the voice at all. Given only `age`, `sex` and `test_time`,
the same model under the row-level split reaches MAE `0.04` and R² `0.996`, better than with all
39 features (MAE `0.61`, R² `0.981`). The voice measures alone reach MAE `5.09` and R² `0.38`.
The scores are interpolated in time between clinic visits, so knowing the patient and the date
nearly fixes the label.

### Baselines

- Predicting the training-fold mean for every recording gives MAE `7.01`. The best pipeline
  (MAE `7.38`, RMSE `9.38`, R² `-0.42`) does not beat it.
- The patient-average baseline reaches MAE `1.75` and R² `0.916` from patient identity alone,
  because scores vary far less within a patient (SD `2.08`) than between patients (SD `7.88`).

### Model Performance

- Best pipeline under honest evaluation: **Extra Trees regressor with Yeo-Johnson**, thresholded
  at 20, giving MAE `7.38` and AUC `0.629`.
- It does not beat plain Extra Trees on AUC (`0.629` against `0.633`), and per-fold AUC ranges
  from `0.798` down to `0.370`, so one held-out patient group is predicted worse than chance.
- Direct binary classification is weaker than regression then threshold, topping out near
  AUC `0.55`. Over 10 further fold assignments the regression route leads in all 10, by AUC
  `0.081` on average, but the 95% interval (`-0.030` to `0.191`) includes zero.
- 3-class macro-F1 sits between `0.27` and `0.37` for every model family, neural networks
  included. The severe class, present in only 6 patients, is the bottleneck.
- Neural networks offer no consistent advantage over the tree ensembles. Huber regression is
  run to convergence and gives MAE `9.93`.
- Per-model numbers for every track are written to `results/*.csv`.

### What the Voice Adds

- On the same five folds, `age`, `sex` and `test_time` alone give AUC `0.634`, all 39 features
  `0.629`, and the voice measures alone `0.520`.
- Over 50 folds (10 further fold assignments), voice only minus all features is `-0.124`
  (95% CI `-0.330` to `0.081`), and demographics minus all is `0.024` (95% CI `-0.021` to `0.069`).

### Patients versus Recordings

- Keeping every training patient and capping their recordings, AUC rises from `0.56` at 10
  recordings per patient to `0.63` at 60, then stays flat.
- Holding the training volume at 800 recordings and varying the number of patients, AUC rises
  from `0.51` with 8 patients to `0.61` with 33. At about 2,000 recordings it rises from `0.52`
  to `0.63`.
- The gain is of the same order as the spread between random draws of 8 patients (SD `0.08`),
  so it shows a direction rather than a precise effect.

### Patient-Aggregate Features: a Negative Result

- Appending the 80-column patient-aggregate block (119 features total) **hurts** performance,
  giving AUC `0.459` to `0.489` against `0.553` for the 39-feature baseline. For the Extra Trees
  regressor, AUC falls from `0.633` to `0.573`.
- Under the row-level split the same block looks like a large improvement (MAE `0.56` to
  `0.08`).
- The aggregates act as a per-patient fingerprint that the trees memorise and cannot transfer
  to unseen patients. This is a common route by which leakage re-enters this benchmark.

### Calibration with a Patient's Own Recordings

- Shifting the predictions by the mean error on `k` of a new patient's recordings with known
  scores lowers MAE from `7.30` to `2.94` with one recording and `2.20` with 20, when those
  recordings are drawn at random from the six months.
- With the patient's first recordings in time, as at the start of monitoring, MAE falls only to
  about `3.6`.
- Predicting the mean of the same calibration scores, with no model, does better at every `k`.

### Minority-Class Oversampling

- SMOTE inside each training fold raises severe-class F1 from `0.150` to `0.216` and macro-F1
  from `0.369` to `0.390`. It adds synthetic recordings, not the extra patients the task needs.

### Hyperparameter Provenance

- The search that produced the values in `models.py` **was not recorded**, and has not been
  reconstructed after the fact. `python tune.py provenance` states what is and is not known.
- It makes little difference: a 16-point sensitivity sweep moves MAE by `0.139` and AUC by
  `0.020` across the whole grid, against `6.77` and `0.369` for the splitter choice, and the
  shipped configuration ranks 7th of 16 on both metrics. Nested cross-validation gives MAE
  `7.376` against `7.384` non-nested, so selection optimism is negligible.
- Tuning the cut-off on inner folds does not help either: the chosen cut-offs range from `13.5`
  to `23.5` and lower macro-F1 from `0.539` to `0.471`, so the clinical cut-off of 20 is kept.

### Conclusion

Once patients are held out properly, the voice features in this dataset support only weak
severity prediction for an unseen patient, around AUC `0.63`, and add little beyond age. The
near-perfect scores reported elsewhere are a property of the evaluation protocol, not of the
models, and can be reached without any voice feature. Any result on this benchmark that does
not state its splitter should be read as the row-level figure.

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
  - Hyperparameter provenance, the sensitivity sweep, nested cross-validation, and threshold
    tuning. Not on the default path and not required to reproduce any reported number.

- `neural.py`
  - PyTorch models: MLP, deeper MLP, and an FT-Transformer-lite feature-attention model.
    Writes `results/results_neural.csv`. Optional, requires `torch`.

- `analyses.py`
  - Supporting analyses: dataset description, baselines, the row-level split with and without
    voice, regression against classification, feature groups, patients versus recordings,
    calibration, and SMOTE.

- `figures.py`
  - Draws the paper's four figures from the CSVs in `results/`.

- `parkinsons_leakage_analysis.ipynb`
  - Single self-contained notebook covering the full exploratory analysis, including the
    patient-mean oracle, a diversity ablation, the aggregation negative result, and few-shot
    personalisation. Committed with outputs. The scripts above are the reference for reported
    numbers; the notebook's Huber, patients-versus-recordings and SMOTE cells use earlier
    settings.

- `tests/test_folds.py`
  - Asserts that folds are patient-disjoint, that transforms are fitted on training rows only,
    that the threshold search stays inside the training patients, and that the patient-average
    baseline never sees the recording it predicts. Run in CI on every push.

- `data/raw/`
  - The dataset and its UCI variable descriptions.

- `results/` and `figures/`
  - Generated by the scripts. Not committed.

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

5. Inspect hyperparameter provenance, selection optimism and threshold tuning:

   ```bash
   python tune.py provenance
   python tune.py sensitivity
   python tune.py nested
   python tune.py threshold
   ```

6. Run the neural models (optional):

   ```bash
   pip install torch
   python neural.py mlp3
   python neural.py mlpdeep
   python neural.py attn
   python neural.py mlp2
   ```

7. Run the supporting analyses and draw the figures. `analyses.py all` takes over an hour on 4
   cores; sections can also be run one at a time (`describe`, `baselines`, `rowlevel`,
   `scoreclass`, `features`, `patients`, `calibration`, `smote`). `figures.py` needs
   `run.py leakage-demo` and `analyses.py all` first:

   ```bash
   python analyses.py all
   python figures.py
   ```

8. Run the tests:

   ```bash
   pip install pytest
   python -m pytest tests/ -v
   ```

Results are written to `results/`, one CSV per section, and figures to `figures/`. The tree
ensembles use 500 to 700 trees and are CPU-heavy, so run sections individually rather than
`all` if you are core-limited.

Reported numbers were produced with Python 3.14.6 and the exact versions pinned in
`requirements.txt`.
