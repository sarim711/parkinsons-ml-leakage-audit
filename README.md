# Parkinson's Telemonitoring — Severity Models

Code for the severity-classification and severity-regression models on the UCI Parkinson's
Telemonitoring dataset, with all preprocessing.

**The central claim of this repository is methodological.** Published results on this benchmark
are routinely reported under row-level cross-validation, which scatters a patient's ~140
recordings across train and test. A model can then identify the *patient* instead of reading the
*voice signal*, and the patient's severity score is nearly constant. Under patient-disjoint
evaluation the same pipeline, features, hyperparameters and seed perform far worse. Section
[Leakage demonstration](#leakage-demonstration) measures the gap directly; it is the number this
repository exists to report.

## Layout

| File | Purpose |
|------|---------|
| `preprocessing.py` | data loading, 39-feature engineering, patient-aggregated features, patient-disjoint folds |
| `models.py` | model factory functions (classical, gradient-boosted, regressors) |
| `evaluate.py` | cross-validated evaluation: classification and regression-then-threshold |
| `run.py` | orchestrator — runs a section, prints metrics, writes CSV/JSON |
| `neural.py` | PyTorch models (MLP, deeper MLP, FeatureAttention) — optional |
| `tune.py` | hyperparameter provenance, nested CV and threshold tuning — **not** on the default path |
| `analyses.py` | supporting analyses in the paper: dataset description, baselines, the split by recording with and without voice, regression versus classification, feature groups, patients vs recordings, calibration, SMOTE |
| `figures.py` | draws the paper's four figures from the CSVs in `results/` |
| `parkinsons_leakage_analysis.ipynb` | archived exploratory notebook, stored without outputs; the scripts above are the reference (see the note at its top) |
| `tests/` | fold-disjointness, train-only-transform and baseline tests (pytest) |
| `.github/workflows/tests.yml` | CI: runs pytest on push and pull request |
| `data/raw/parkinsons_updrs.data` | dataset, committed (see attribution below) |
| `data/raw/parkinsons_updrs.names` | dataset variable descriptions, as distributed by UCI |
| `requirements.txt` | pinned dependency versions |
| `LICENSE`, `CITATION.cff` | license and citation metadata |

`results/` and `figures/` are **not** committed. They are created on demand by `run.py`, `tune.py`,
`neural.py`, `analyses.py` and `figures.py` and are listed in `.gitignore`, because every file in them is
regenerable from the committed code and data. Run a section to produce them.

## Install

```bash
pip install -r requirements.txt   # pinned versions used to produce the results
pip install torch                 # only for neural.py
pip install pytest                # only to run the tests
```

Versions are pinned because tree-ensemble and gradient-boosting libraries
(scikit-learn, XGBoost, LightGBM) can shift metrics slightly across releases. Use the
pinned versions to reproduce the reported numbers exactly.

## Dataset

UCI Parkinson's Telemonitoring (5,875 voice recordings from 42 early-stage patients).
`data/raw/parkinsons_updrs.data` is committed for convenience and is also available from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring)
(doi:10.24432/C5ZS3N), which distributes it under a CC BY 4.0 license. The MIT license of this
repository covers the code only. If you use the data, please cite the original authors:

> A. Tsanas, M. A. Little, P. E. McSharry, L. O. Ramig (2010). *Accurate telemonitoring of
> Parkinson's disease progression by non-invasive speech tests.* IEEE Transactions on
> Biomedical Engineering.

## Pipeline summary

**Labels** (from `motor_UPDRS`):
- 3-class — Mild `<20`, Moderate `[20,30)`, Severe `>=30`
- Binary — Not-Mild `>=20` vs Mild `<20`

**Features (39):** 3 demographics (`age`, `sex`, `test_time`) + 16 raw voice measures +
11 log-transformed jitter/shimmer variants + 4 ratios + 5 age interactions. The
patient-aggregation experiments append per-patient mean/std/min/max/median of the 16 raw
voice features (80 columns) for 119 features total.

**Cross-validation:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`,
grouped by `subject_id` so no patient appears in both train and test. The stratification
target is the per-patient majority of the binary label for the binary/regression tracks and
of the 3-class label for the 3-class track. `evaluate.py` asserts this invariant on every fold
at run time; `tests/test_folds.py` asserts it independently.

**Preprocessing inside each fold** (fit on train, applied to test): optional standardization
(linear models), optional Yeo-Johnson power transform, optional PCA.

> Note: Ridge, Huber and SVR are never passed `scale=True`. They are standardized *implicitly*,
> because `PowerTransformer` defaults to `standardize=True` and those three models are the ones
> run with the Yeo-Johnson transform enabled.

**Class balancing:** `class_weight='balanced'` / `scale_pos_weight` / balanced `sample_weight`
depending on the estimator.

**Regression-then-threshold:** regressors predict continuous `motor_UPDRS`; predictions are
thresholded at 20 to recover the binary label. Metrics are mean ± sample std (ddof=1) over folds.

## Leakage demonstration

`python run.py leakage-demo` holds everything fixed — the same 39 features, the same
`ExtraTreesRegressor(n_estimators=700, min_samples_leaf=2)`, the same Yeo-Johnson transform, the
same threshold of 20, the same seed 42 — and changes **only the splitter**:

| Protocol | MAE | AUC | Accuracy | Macro-F1 | Balanced acc |
|---|---|---|---|---|---|
| Row-level `StratifiedKFold` (leaky) | 0.61 ± 0.04 | 0.998 ± 0.001 | 0.982 | 0.982 | 0.982 |
| Patient-disjoint `StratifiedGroupKFold` | 7.38 ± 1.69 | 0.629 ± 0.183 | 0.562 | 0.539 | 0.566 |
| **Difference** | **−6.77** | **+0.369** | **+0.421** | **+0.443** | **+0.416** |

Under the row-level protocol **all 42 patients appear in both train and test in every fold**
(the section prints this count); under the patient-disjoint protocol the count is 0.

The first row is not a better model. It is the same model scoring its own ability to recognise
patients it has already seen — and it lands squarely in the range reported by published work on
this benchmark (near-perfect accuracy, R²/AUC ≳ 0.99). The second row is what the voice
features actually support for a patient the model has never seen. **Any result on this dataset
that does not state its splitter should be read as the first row.**

**The row-level score does not need the voice at all** (`python analyses.py rowlevel`). Same
row-level folds, same pipeline, different feature sets:

| Features given to the model (row-level split) | MAE | R² | AUC |
|---|---|---|---|
| All 39 features | 0.61 | 0.981 | 0.998 |
| **Age, sex and test time only (no voice)** | **0.04** | **0.996** | **1.000** |
| Voice measures only (31 features) | 5.09 | 0.383 | 0.823 |

The motor-UPDRS labels are linearly interpolated in time between clinic visits, so once a model can
tell which patient a recording belongs to (age and sex nearly do that) and when it was made, the label
is almost determined. Three non-voice columns reach R² 0.996 under the row-level split —
better than the full feature set — while the voice measures alone do far worse.

## Results

All numbers below are patient-disjoint, 5-fold, mean ± sample std (ddof=1), produced by actually
running the code in this repository. Reproduce with `python run.py all`.

> Environment: Python 3.14.6 with the exact versions in `requirements.txt`
> (numpy 2.4.6, pandas 3.0.3, scipy 1.18.0, scikit-learn 1.9.0, xgboost 3.3.0, lightgbm 4.7.0).
> The pins and these tables were produced by the same environment.

**Regression → threshold 20** (`python run.py regression`)

| Model | MAE | Accuracy | Macro-F1 | Balanced acc | AUC |
|---|---|---|---|---|---|
| Ridge | 9.41 | 0.479 | 0.469 | 0.478 | 0.464 |
| Huber | 9.93 | 0.470 | 0.460 | 0.472 | 0.448 |
| SVR (RBF) | 8.23 | 0.479 | 0.465 | 0.480 | 0.490 |
| ExtraTrees | 7.41 | 0.561 | 0.539 | 0.565 | 0.633 |
| **ExtraTrees + Yeo-Johnson** | **7.38** | **0.562** | **0.539** | **0.566** | **0.629** |
| RandomForest | 8.07 | 0.526 | 0.500 | 0.529 | 0.613 |
| XGBoost | 7.98 | 0.537 | 0.523 | 0.537 | 0.598 |
| LightGBM | 8.24 | 0.490 | 0.465 | 0.500 | 0.591 |
| GradientBoosting | 8.30 | 0.499 | 0.475 | 0.503 | 0.581 |
| ExtraTrees + patient summaries (119 features) | 7.21 | 0.529 | 0.516 | 0.535 | 0.573 |

> **Huber** is run with `max_iter=10000`. At the earlier limit of 500, lbfgs stopped unconverged in all five
> folds (MAE 9.88, AUC 0.456); the converged fit above is marginally worse.
>
> An `XGBoost (pseudo-Huber)` variant was previously included here and has been **removed**: with
> `objective="reg:pseudohubererror"` on the unscaled 0–40 target it diverged, producing an MAE of
> ~2660 and an AUC of exactly 0.500. It was a broken configuration rather than a weak model, and
> reporting it alongside working baselines was misleading.

The last row adds the 80 per-patient summary columns to plain ExtraTrees; see *Literature-driven
variants* below for what it shows.

The headline pipeline is ExtraTrees + Yeo-Johnson, but note it does not beat plain ExtraTrees on
AUC (0.629 vs 0.633), and its fold-to-fold spread is very wide — per-fold AUC ranges from **0.798
down to 0.370**, i.e. one of five held-out patient groups is predicted worse than chance. An
average AUC of 0.63 with that spread is a weak signal, not a working clinical classifier.

**Binary classification, direct** (`python run.py 2class`)

| Model | Accuracy | Macro-F1 | Balanced acc | AUC |
|---|---|---|---|---|
| LogisticRegression | 0.448 | 0.442 | 0.452 | 0.421 |
| RandomForest | 0.478 | 0.457 | 0.479 | 0.482 |
| ExtraTrees | 0.564 | 0.544 | 0.565 | 0.553 |
| XGBoost | 0.525 | 0.508 | 0.534 | 0.525 |
| LightGBM | 0.499 | 0.476 | 0.509 | 0.522 |

Direct binary classification tops out around AUC 0.55, below the regression-then-threshold route.
Over ten further fold assignments (`python analyses.py scoreclass`) the final pipeline is ahead of the
direct ExtraTrees classifier in 10 of 10, by a mean AUC of +0.081, but the corrected resampled t interval includes zero (95% CI −0.030 to +0.191).

**3-class severity** (`python run.py 3class`)

| Model | Accuracy | Macro-F1 | Balanced acc | F1 (severe) |
|---|---|---|---|---|
| LogisticRegression | 0.323 | 0.293 | 0.314 | 0.186 |
| RandomForest | 0.441 | 0.371 | 0.384 | 0.154 |
| XGBoost | 0.411 | 0.342 | 0.372 | 0.163 |
| LightGBM | 0.417 | 0.341 | 0.362 | 0.121 |
| ExtraTrees | 0.434 | 0.323 | 0.346 | 0.026 |

Macro-F1 sits at 0.29–0.37 across every model family. The severe class is the bottleneck.

**Neural models** (`python neural.py mlp3|mlpdeep|attn|mlp2`, optional, need `torch`)

| Model | Task | Accuracy | Macro-F1 | Balanced acc | F1 (severe) | AUC |
|---|---|---|---|---|---|---|
| MLP, two hidden layers | 3-class | 0.383 | 0.360 | 0.376 | 0.313 | – |
| MLP, three hidden layers | 3-class | 0.388 | 0.355 | 0.363 | 0.245 | – |
| FeatureAttention | 3-class | 0.309 | 0.269 | 0.285 | 0.210 | – |
| MLP, two hidden layers | binary | 0.522 | 0.510 | 0.518 | – | 0.558 |

The networks do not beat the tree ensembles: the 3-class macro-F1 stays below RandomForest's, and the binary MLP's AUC
is level with the direct ExtraTrees classifier.

**Literature-driven variants** (`python run.py literature`, binary task)

| Variant | Accuracy | Macro-F1 | Balanced acc | AUC |
|---|---|---|---|---|
| A: Yeo-Johnson + ExtraTrees | 0.554 | 0.535 | 0.554 | 0.536 |
| B: Yeo-Johnson + PCA(20) + ExtraTrees | 0.490 | 0.478 | 0.488 | 0.497 |
| C1: patient-aggregates + LightGBM | 0.435 | 0.414 | 0.450 | 0.489 |
| C2: patient-aggregates + ExtraTrees | 0.439 | 0.435 | 0.443 | 0.459 |
| C3: patient-aggregates + XGBoost | 0.469 | 0.452 | 0.484 | 0.460 |
| D: patient-aggregates + Yeo-Johnson + ExtraTrees | 0.462 | 0.448 | 0.467 | 0.461 |
| E: patient-aggregates + Yeo-Johnson + LightGBM | 0.444 | 0.437 | 0.459 | 0.473 |

Appending per-patient aggregate features (rows C–E, 119 features) **hurts** relative to the
39-feature baseline (AUC 0.459–0.489 versus 0.553 for plain ExtraTrees). The aggregates act as a
per-patient fingerprint that the trees memorise and cannot transfer to unseen patients — a
negative result, and a place where leakage tends to re-enter this benchmark unnoticed. The regressor shows the
same thing (last row of the regression table: AUC 0.633 → 0.573), while under the row-level split
the aggregates look like a large improvement (MAE 0.56 → 0.08, `python analyses.py rowlevel`):
a representation that suits leakage passes for a useful feature.

**Hyperparameter sensitivity** (`python tune.py sensitivity`)

A 16-point grid around the shipped values, same pipeline, same five patient-disjoint folds:

| | MAE range | AUC range |
|---|---|---|
| Across all 16 configurations | 7.341 – 7.480 (spread **0.139**) | 0.614 – 0.634 (spread **0.020**) |
| Shipped `models.py` config | 7.384 (rank 7/16) | 0.629 (rank 7/16) |
| *For comparison: the leakage effect* | *6.77* | *0.369* |

The entire hyperparameter grid moves MAE by 0.14 and AUC by 0.02. The splitter choice moves them
by 6.77 and 0.369 — roughly **50×** and **18×** larger. The shipped configuration sits mid-table
on both metrics (7th of 16), not at the top, which is what you would expect from values that were
*not* cherry-picked against the reporting folds. So while the original search is undocumented, no
choice within this range could have produced the reported result; the missing record is a
documentation gap, not a validity threat.

**Nested cross-validation** (`python tune.py nested`)

| Protocol | MAE | AUC | Accuracy | Macro-F1 | Balanced acc |
|---|---|---|---|---|---|
| Non-nested (fixed `models.py`, as reported) | 7.384 ± 1.689 | 0.629 ± 0.183 | 0.562 | 0.539 | 0.566 |
| Nested (inner selection per outer fold) | 7.376 ± 1.676 | 0.628 ± 0.183 | 0.555 | 0.533 | 0.559 |
| Selection optimism | +0.01 | +0.001 | +0.006 | +0.006 | +0.007 |

**Selection optimism is negligible here.** The inner search chose a configuration *different* from
`models.py` in all five outer folds, yet outer-fold performance barely moved. The pipeline is
insensitive to these hyperparameters in this range, so the reported numbers are not inflated by
hyperparameter choice — the weak performance is a property of the task under patient-disjoint
evaluation, not an artefact of tuning. This does not substitute for recording the original search;
see [Hyperparameter selection](#hyperparameter-selection).

**Threshold tuning** (`python tune.py threshold`)

The cut-off applied to the predicted score is chosen per outer fold, from 10 to 30 in steps of 0.5,
by maximising macro-F1 on inner patient-disjoint out-of-fold predictions of the training patients.
The outer test patients take no part in the choice.

| Protocol | Accuracy | Macro-F1 | Balanced acc |
|---|---|---|---|
| Fixed cut-off 20 (as reported) | 0.562 | 0.539 | 0.566 |
| Cut-off tuned on inner folds | 0.537 | 0.471 | 0.514 |

**Tuning the cut-off does not help.** The inner search selected 15.0, 15.5, 13.5, 23.5 and 15.5 in
the five folds, and the tuned cut-off scored worse than 20 in four of them. A cut-off chosen on
one set of patients does not carry over to new patients, so the fixed clinical cut-off of 20 is
kept. MAE and AUC do not depend on the cut-off and are unchanged.


**Supporting analyses** (`python analyses.py all`)

All patient-disjoint, on the reporting folds unless stated.

| Analysis | Result |
|---|---|
| Dataset description (`describe`) | cohort, class counts per recording and per patient, and Pearson r with motor UPDRS (age 0.27, PPE 0.16, HNR −0.16, shimmer APQ11 0.14), written to `results/dataset_description.csv` |
| Overall-average baseline (training-fold mean) | MAE 7.01, RMSE 8.12; the final pipeline (MAE 7.38, RMSE 9.38, R² −0.42) does not beat it |
| Pooled out-of-fold AUC of the final pipeline | 0.575 (mean of per-fold AUCs: 0.629) |
| Patient-average baseline (no voice features) | MAE 1.75, R² 0.916; within-patient SD 2.08 vs between-patient SD 7.88 |
| Feature groups (AUC) | all 0.629 · age, sex, test time 0.634 · age and sex 0.618 · voice only 0.520 · without test time 0.627 |
| AUC differences over 10 further fold assignments (50 folds, corrected resampled t) | demographics − all: +0.024 (95% CI −0.021 to 0.069); voice − all: −0.124 (95% CI −0.330 to 0.081) |
| Recordings per patient (all training patients kept) | 10 → all recordings: AUC 0.559 → 0.628, flat from about 60 |
| Patients at a fixed volume of ~800 recordings (5 folds, 20 draws) | 8 / 16 / 24 / 33 patients: AUC 0.509 / 0.559 / 0.564 / 0.610; MAE 8.44 → 7.30 |
| Patients at a fixed volume of ~2,000 recordings (5 folds, 20 draws) | 8 / 16 / 24 / 33 patients: AUC 0.521 / 0.574 / 0.601 / 0.627; MAE 8.29 → 7.33 (8-patient arm has only ~1,100 recordings) |
| Calibration, random k recordings | MAE 7.30 → 2.94 (k=1) → 2.20 (k=20) |
| Calibration, first k recordings in time | MAE 7.30 → 3.54 (k=1), then flat at 3.58–3.64 |
| SMOTE (three-class, random forest, features standardized first) | severe-class F1 0.150 → 0.216, macro-F1 0.369 → 0.390 (Borderline-SMOTE: 0.213, 0.390) |

The fixed-volume experiment draws 8, 16, 24 or 33 of each fold's training patients at random and caps their
recordings so the training volume stays the same. At ~800 recordings every arm reaches the target volume, so
that row is the clean equal-volume comparison. More patients help, by 0.10–0.11 AUC, which is of the same order as
the spread between random draws of eight patients (SD 0.08); more recordings per patient stop helping at about 60.

Random calibration is optimistic: it lets recordings from later in the trial, whose interpolated scores lie
close to those being predicted, inform the offset. The first-k version is what a clinic would have.

## Hyperparameter selection

The hyperparameters are **fixed literals in `models.py`** (for example the final pipeline is
`ExtraTreesRegressor(n_estimators=700, min_samples_leaf=2, random_state=42)`). Nothing is searched
at run time, so `run.py` reproduces the reported numbers exactly on every invocation.

**The search that produced those literals was not recorded, and has not been reconstructed.**
`models.py` contains the selected values only — no grid, no search script, no log — and no record
survives elsewhere in the repository. `tune.py` states this as an explicit declaration rather than
guessing a plausible grid after the fact, which would amount to a fabricated methods section. Run
`python tune.py provenance` for the full statement of what is and is not known.

Because the selection procedure is unknown, it cannot be ruled out that it used the same five
patient-disjoint folds later used for reporting — which would make those folds no longer held out
with respect to the hyperparameter choice. Two measurements bound what that could cost:

- **`python tune.py sensitivity`** sweeps a 16-point grid (`n_estimators` × `min_samples_leaf`)
  around the shipped values on the reporting folds, showing how far the score can move at all.
- **`python tune.py nested`** performs the selection *inside* each outer fold — five
  patient-disjoint outer folds for reporting, an inner `StratifiedGroupKFold` over training
  patients only choosing among four configurations (one being exactly the `models.py` values).

Both are reported under [Results](#results), and both find the effect negligible. The practical conclusion is that
the missing search record is a documentation gap, not a threat to the reported numbers: the final
pipeline is insensitive to these hyperparameters across the range tested, so no choice within that
range could have manufactured the result. `tune.py` is not imported by `run.py` and is not required
to reproduce anything.

## Run

```bash
python run.py regression-final     # ExtraTreesRegressor + Yeo-Johnson, per-fold table + OOF confusion
python run.py leakage-demo         # row-level vs patient-disjoint splitting (the central claim)
python run.py 3class               # 3-class classifiers
python run.py 2class               # binary classifiers
python run.py literature           # Yeo-Johnson / PCA / patient-aggregation variants
python run.py regression           # all regression-then-threshold models
python run.py all                  # everything (slow)

# hyperparameter provenance, sensitivity, nested CV and threshold tuning (optional, off the default path)
python tune.py provenance
python tune.py sensitivity
python tune.py nested
python tune.py threshold

# neural models (optional, require torch); each adds its row to results/results_neural.csv
python neural.py mlp2
python neural.py mlp3
python neural.py mlpdeep
python neural.py attn

# supporting analyses and figures in the paper (analyses.py all takes over an hour on 4 cores)
python analyses.py all             # or one of: describe baselines rowlevel scoreclass
                                   #            features patients calibration smote
python figures.py                  # needs run.py leakage-demo and analyses.py all first
```

Results are written under `results/` (one CSV per section; the final pipeline also writes a
JSON with the per-fold table and pooled out-of-fold confusion matrix). Pass `--out <dir>` to any
script to write somewhere else.

Note: the 500/700-tree ensembles are CPU-heavy. On a single core each section takes several
minutes; run sections individually rather than `all` if you are core-limited.

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

`tests/test_folds.py` asserts the two invariants every reported number depends on:

1. **Patient-disjoint folds** — for both the binary and 3-class stratification targets, the
   intersection of train and test subject ids is empty in every fold; every recording is tested
   exactly once; every subject is held out exactly once. A further test confirms the guard in
   `evaluate.py` actually *fires* when handed a row-level split.
2. **Train-only transforms** — the Yeo-Johnson transform is fitted on training rows only. This is
   checked behaviourally (perturbing the test rows must not move the transformed training rows)
   and by comparing against a `PowerTransformer` fitted on the training rows alone. The same
   check covers `StandardScaler` and PCA.

It also checks that the inner folds used for threshold tuning in `tune.py` stay inside the training
patients, and that the patient-average baseline never uses the recording it predicts.

CI runs the suite on every push and pull request via `.github/workflows/tests.yml`.

## Reproducibility notes

- Fixed seed (42) for all models and folds.
- Patient-disjoint folds: no subject appears in both train and test, asserted at run time.
- Metrics are mean ± sample standard deviation (ddof=1) across the 5 folds.
- Hyperparameters are fixed literals in `models.py`; no search runs on the default path.
- The `results/` and `figures/` directories are regenerated by the scripts and are git-ignored.

## Citing this repository

See `CITATION.cff`; GitHub renders a "Cite this repository" button from it.
