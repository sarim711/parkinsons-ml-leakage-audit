# Legacy — superseded XAI / adversarial-robustness direction

This directory holds the original direction of the project, kept for the record. It is **not**
part of the current pipeline and nothing in the repository root imports it.

## What this was

A regression pipeline for Parkinson's severity on the Oxford/UCI Telemonitoring dataset, intended
to add Explainable AI (SHAP / LIME) for feature importance and the Adversarial Robustness Toolbox
(ART) for stability testing — specifically, perturbing the voice features to measure the effect on
model performance.

| File | What it is |
|------|------------|
| `01_preprocessing.ipynb` | preprocessing notebook |
| `preprocessing_1.ipynb` | Colab preprocessing notebook |
| `base.yaml` | original config: target `total_UPDRS`, group column `subject#`, 80/20 split |
| `load_data.py` | config-driven raw data loader |
| `requirements-legacy.txt` | dependencies for the above (ART, SHAP, LIME, PyYAML) |

## Why it was set aside

The perturbation study needed a trustworthy baseline to perturb *against*, and establishing one
surfaced a prior problem: results reported on this benchmark are not evaluated under a consistent
protocol, and the common row-level split lets a model identify the patient rather than read the
voice signal. Measuring robustness on top of a number produced that way would inherit the flaw
rather than test it.

So the question changed from *"how stable is this model under perturbation?"* to *"what does this
benchmark actually support once patients are held out properly?"*

The thread was already visible in `base.yaml`, which sets `group_column: "subject#"` with the note
*"The column used to split train/test (to prevent data leakage)"* — the current work follows that
observation through and quantifies what ignoring it costs. See the root `README.md`.

## Note on the target variable

The original direction predicted `total_UPDRS`. The current work predicts `motor_UPDRS`, and
derives the severity classes from it.
