"""
Hyperparameter provenance and nested cross-validation.

This module is NOT part of the default path. Nothing in run.py imports it, and
none of the reported results depend on it. The hyperparameters used for the
reported numbers are the literals in models.py; this file exists to document
where they came from and to quantify how optimistic the reported numbers are as
a consequence of having selected them.

Usage:
    python tune.py provenance    # what is and is not known about the selection
    python tune.py sensitivity   # how much the choice actually matters (~6 min)
    python tune.py nested        # nested CV for the final pipeline (~4 min)

Three different things live here, and they must not be confused:

  * RECORDED_SEARCH -- the search that actually produced the values in models.py.
    Status: NOT RECORDED. models.py contains only the selected literals, with no
    grid, no search script and no log, and no record survives elsewhere in the
    repository. It is declared as unrecorded rather than reconstructed: a
    plausible-looking grid that was not the one actually run would be a
    fabricated methods section.

  * SENSITIVITY_GRID -- a retrospective sweep, defined and run HERE, measuring
    how much the final pipeline's score moves across a wide range of the two
    hyperparameters in question. This does not recover the original search. It
    bounds how much that search could possibly have mattered.

  * NESTED_CANDIDATES -- a small grid for the nested-CV estimate only. It is not
    a claim about what was originally searched.
"""
import os
import sys
import textwrap

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             roc_auc_score)

import preprocessing as P
import models as M
import evaluate as E

USAGE = "usage: python tune.py {provenance|sensitivity|nested} [--out results]"


# --------------------------------------------------------------------------
# 1. Provenance of the values in models.py
# --------------------------------------------------------------------------
# The hyperparameters in models.py were not accompanied by a search record.
# This is a positive declaration of that fact, not an unfilled placeholder: the
# grid, strategy, scoring metric and selection splitter are unknown, were not
# reconstructed, and must not be cited as if they were known.
#
# What replaces them, for anyone assessing whether the reported numbers can be
# trusted, is direct measurement of the consequence:
#
#   `python tune.py sensitivity`  -- spread of the final pipeline's score across
#                                    a 16-point grid around the shipped values.
#   `python tune.py nested`       -- selection optimism, by selecting inside each
#                                    outer fold instead of once, globally.
#
# Both are reported in README.md. If the original search notes ever surface,
# replace the declaration below with the real record.
RECORDED_SEARCH = {
    "status": "NOT_RECORDED",
    "grid": None,
    "strategy": None,
    "scoring": None,
    "selection_splitter": None,
    "date": None,
    "declaration": (
        "The search that produced the hyperparameters in models.py was not "
        "recorded. models.py contains the selected literals only; no grid, "
        "search script or log exists in this repository, and none was "
        "reconstructed after the fact. The hyperparameters are therefore "
        "fixed constants of the reported pipeline, not the output of a "
        "documented selection procedure. Because the selection procedure is "
        "unknown, it cannot be ruled out that it used the same five "
        "patient-disjoint folds later used for reporting. The `sensitivity` "
        "and `nested` commands quantify what that could cost; both find the "
        "effect to be negligible."
    ),
}

# Which model factories the reported pipeline actually depends on. The selected
# values are read back from models.py so this listing cannot drift from the code.
_PROVENANCE_FACTORIES = [
    ("reg_extra_trees (final pipeline)", M.reg_extra_trees),
    ("reg_random_forest", M.reg_random_forest),
    ("reg_xgboost", M.reg_xgboost),
    ("reg_lightgbm", M.reg_lightgbm),
    ("reg_gradient_boosting", M.reg_gradient_boosting),
    ("clf_extra_trees", M.clf_extra_trees),
    ("clf_random_forest", M.clf_random_forest),
    ("clf_lightgbm", M.clf_lightgbm),
    ("clf_xgboost", M.clf_xgboost),
]

# Parameters worth showing (the rest are library defaults).
_SHOW = ("n_estimators", "min_samples_leaf", "max_depth", "learning_rate",
         "num_leaves", "colsample_bytree", "subsample", "alpha", "epsilon", "C")


def print_provenance():
    print("=== Selected values (read back from models.py) ===")
    for label, factory in _PROVENANCE_FACTORIES:
        params = factory().get_params()
        shown = {k: params[k] for k in _SHOW if k in params and params[k] is not None}
        print(f"  {label:34s} {shown}")

    print("\n=== Provenance of those values ===")
    if RECORDED_SEARCH["status"] == "NOT_RECORDED":
        print("  status: NOT RECORDED\n")
        for line in textwrap.wrap(RECORDED_SEARCH["declaration"], width=76):
            print(f"  {line}")
        print("\n  Unknown, and deliberately not reconstructed:")
        for k in ("grid", "strategy", "scoring", "selection_splitter", "date"):
            print(f"    - {k}")
        print("\n  Measured instead:")
        print("    python tune.py sensitivity   # does the choice move the score?")
        print("    python tune.py nested        # selection optimism on the reported folds")
    else:
        for k, v in RECORDED_SEARCH.items():
            print(f"    - {k}: {v}")


# --------------------------------------------------------------------------
# 2. Nested cross-validation for the final pipeline only
# --------------------------------------------------------------------------
# Small, explicit grid for the nested estimate. Index 2 is exactly the
# configuration hard-coded in models.py:reg_extra_trees, so the inner search is
# able to recover the reported pipeline.
NESTED_CANDIDATES = [
    {"n_estimators": 300, "min_samples_leaf": 1},
    {"n_estimators": 300, "min_samples_leaf": 2},
    {"n_estimators": 700, "min_samples_leaf": 2},   # == models.py reg_extra_trees
    {"n_estimators": 700, "min_samples_leaf": 4},
]
N_INNER_SPLITS = 3
SELECTION_METRIC = "mae"   # inner selection minimises MAE, the regressor's own loss


def _make_candidate(cfg):
    """Same estimator class and seed as models.py:reg_extra_trees, varying cfg only."""
    return ExtraTreesRegressor(random_state=P.SEED, n_jobs=-1, **cfg)


def _inner_select(Xv, y_reg, y2, groups, tr):
    """Choose a config using only the training patients of one outer fold.

    Inner splitter is StratifiedGroupKFold grouped by subject_id, so the
    selection itself is patient-disjoint. No outer-test row is touched.
    """
    g_tr = groups[tr]
    strat_tr = P._patient_majority(y2[tr], g_tr)
    sgkf = StratifiedGroupKFold(n_splits=N_INNER_SPLITS, shuffle=True, random_state=P.SEED)
    inner = list(sgkf.split(np.zeros(len(tr)), strat_tr, g_tr))
    for j, (itr, ite) in enumerate(inner):
        assert not np.intersect1d(g_tr[itr], g_tr[ite]).size, f"inner fold {j} leaks patients"

    scores = []
    for cfg in NESTED_CANDIDATES:
        maes = []
        for itr, ite in inner:
            a, b = E._fold_preprocess(Xv[tr][itr], Xv[tr][ite], yeo_johnson=True)
            model = _make_candidate(cfg)
            model.fit(a, y_reg[tr][itr])
            maes.append(float(np.mean(np.abs(model.predict(b) - y_reg[tr][ite]))))
        scores.append(float(np.mean(maes)))
    return NESTED_CANDIDATES[int(np.argmin(scores))], scores


def run_nested(outdir):
    df = P.load_raw()
    X, y_reg, y3, y2, groups = P.get_xy(df)
    Xv = X.values
    outer = P.make_folds(df, y2, groups)   # the same five reporting folds

    print(f"loaded {len(df)} recordings, {df['subject_id'].nunique()} patients, "
          f"{X.shape[1]} features")
    print(f"\n=== Nested CV: 5 patient-disjoint outer folds x "
          f"{N_INNER_SPLITS} inner patient-disjoint folds x "
          f"{len(NESTED_CANDIDATES)} configs ===")
    print(f"inner selection minimises {SELECTION_METRIC}; outer folds are the "
          f"same ones used for the reported numbers")

    rows, mae, auc, acc, f1m, bacc = [], [], [], [], [], []
    for k, (tr, te) in enumerate(outer):
        assert not np.intersect1d(groups[tr], groups[te]).size, f"outer fold {k} leaks patients"
        best, scores = _inner_select(Xv, y_reg, y2, groups, tr)

        a, b = E._fold_preprocess(Xv[tr], Xv[te], yeo_johnson=True)
        model = _make_candidate(best)
        model.fit(a, y_reg[tr])
        pred = model.predict(b)

        yt = (y_reg[te] >= E.THRESHOLD).astype(int)
        yh = (pred >= E.THRESHOLD).astype(int)
        f_mae = float(np.mean(np.abs(pred - y_reg[te])))
        f_auc = roc_auc_score(yt, pred) if len(np.unique(yt)) > 1 else 0.5
        f_acc = accuracy_score(yt, yh)
        f_f1 = f1_score(yt, yh, average="macro")
        f_bacc = balanced_accuracy_score(yt, yh)
        mae.append(f_mae); auc.append(f_auc); acc.append(f_acc)
        f1m.append(f_f1); bacc.append(f_bacc)

        is_reported = best == NESTED_CANDIDATES[2]
        print(f"  fold {k}: selected {best} "
              f"{'(== models.py)' if is_reported else '(DIFFERS from models.py)'}  "
              f"inner {SELECTION_METRIC}={[round(s, 3) for s in scores]}")
        print(f"          outer MAE={f_mae:.2f} AUC={f_auc:.3f} Acc={f_acc:.3f} "
              f"MacroF1={f_f1:.3f} BalAcc={f_bacc:.3f}")
        rows.append({"fold": k, "selected_n_estimators": best["n_estimators"],
                     "selected_min_samples_leaf": best["min_samples_leaf"],
                     "selected_equals_models_py": bool(is_reported),
                     "mae": f_mae, "auc": f_auc, "accuracy": f_acc,
                     "macro_f1": f_f1, "balanced_acc": f_bacc})

    sd = lambda v: float(np.std(v, ddof=1))
    nested = {"mae": float(np.mean(mae)), "mae_std": sd(mae),
              "auc": float(np.mean(auc)), "auc_std": sd(auc),
              "accuracy": float(np.mean(acc)), "macro_f1": float(np.mean(f1m)),
              "balanced_acc": float(np.mean(bacc))}

    # The reported, non-nested number: fixed models.py config on the same outer folds.
    flat = E.evaluate_regression_threshold(X, y_reg, groups, outer, M.reg_extra_trees,
                                           yeo_johnson=True)

    print("\n=== Nested vs non-nested (same outer folds, same seed) ===")
    print(f"  {'non-nested (reported)':24s} MAE={flat['mae']:.2f}±{flat['mae_std']:.2f}  "
          f"AUC={flat['auc']:.3f}±{flat['auc_std']:.3f}  Acc={flat['accuracy']:.3f}  "
          f"MacroF1={flat['macro_f1']:.3f}  BalAcc={flat['balanced_acc']:.3f}")
    print(f"  {'nested':24s} MAE={nested['mae']:.2f}±{nested['mae_std']:.2f}  "
          f"AUC={nested['auc']:.3f}±{nested['auc_std']:.3f}  Acc={nested['accuracy']:.3f}  "
          f"MacroF1={nested['macro_f1']:.3f}  BalAcc={nested['balanced_acc']:.3f}")
    print(f"  {'selection optimism':24s} MAE={flat['mae'] - nested['mae']:+.2f}  "
          f"AUC={flat['auc'] - nested['auc']:+.3f}  "
          f"Acc={flat['accuracy'] - nested['accuracy']:+.3f}")
    print("  (a negative MAE delta / positive AUC delta means the reported number")
    print("   flatters the pipeline relative to selecting inside each fold)")

    os.makedirs(outdir, exist_ok=True)
    per_fold = os.path.join(outdir, "nested_cv_folds.csv")
    pd.DataFrame(rows).to_csv(per_fold, index=False)
    summary = os.path.join(outdir, "nested_cv_summary.csv")
    pd.DataFrame([{"protocol": "non_nested_fixed_models_py",
                   **{k: flat[k] for k in ("mae", "mae_std", "auc", "auc_std",
                                           "accuracy", "macro_f1", "balanced_acc")}},
                  {"protocol": "nested_inner_selection", **nested}]).to_csv(summary, index=False)
    print(f"  -> saved {per_fold}")
    print(f"  -> saved {summary}")
    return nested, flat


# --------------------------------------------------------------------------
# 3. Retrospective sensitivity sweep
# --------------------------------------------------------------------------
# Fully documented, because it is run by this file rather than remembered.
#
#   grid      : the cross product below, 16 configurations
#   estimator : ExtraTreesRegressor, random_state=42, all other params default
#   pipeline  : Yeo-Johnson (fit per fold on train) -> regress -> threshold 20
#   protocol  : the same five patient-disjoint StratifiedGroupKFold folds used
#               for reporting; metrics are the fold means
#   reported  : MAE and AUC per configuration, plus the spread across the grid
#
# This measures how much the hyperparameter choice can move the result. It does
# NOT recover how the shipped values were originally chosen.
SENSITIVITY_GRID = {
    "n_estimators": [100, 300, 700, 1000],
    "min_samples_leaf": [1, 2, 4, 8],
}
SHIPPED = {"n_estimators": 700, "min_samples_leaf": 2}   # == models.py reg_extra_trees


def run_sensitivity(outdir):
    df = P.load_raw()
    X, y_reg, y3, y2, groups = P.get_xy(df)
    folds = P.make_folds(df, y2, groups)

    configs = [{"n_estimators": n, "min_samples_leaf": m}
               for n in SENSITIVITY_GRID["n_estimators"]
               for m in SENSITIVITY_GRID["min_samples_leaf"]]
    print(f"loaded {len(df)} recordings, {df['subject_id'].nunique()} patients, "
          f"{X.shape[1]} features")
    print(f"\n=== Sensitivity sweep: {len(configs)} configurations x 5 patient-disjoint folds ===")
    print("pipeline fixed at Yeo-Johnson -> ExtraTreesRegressor -> threshold 20, seed 42\n")

    rows = []
    for cfg in configs:
        r = E.evaluate_regression_threshold(
            X, y_reg, groups, folds,
            lambda c=cfg: ExtraTreesRegressor(random_state=P.SEED, n_jobs=-1, **c),
            yeo_johnson=True)
        is_shipped = cfg == SHIPPED
        rows.append({**cfg, "is_shipped": is_shipped, "mae": r["mae"], "auc": r["auc"],
                     "accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
                     "balanced_acc": r["balanced_acc"]})
        print(f"  n_estimators={cfg['n_estimators']:5d} min_samples_leaf={cfg['min_samples_leaf']:2d}"
              f"  MAE={r['mae']:.3f}  AUC={r['auc']:.3f}"
              f"{'   <-- shipped in models.py' if is_shipped else ''}")

    frame = pd.DataFrame(rows)
    mae_lo, mae_hi = frame["mae"].min(), frame["mae"].max()
    auc_lo, auc_hi = frame["auc"].min(), frame["auc"].max()
    shipped_row = frame[frame["is_shipped"]].iloc[0]
    mae_rank = int((frame["mae"] < shipped_row["mae"]).sum()) + 1
    auc_rank = int((frame["auc"] > shipped_row["auc"]).sum()) + 1

    print(f"\n  MAE across the grid: {mae_lo:.3f} to {mae_hi:.3f}  (spread {mae_hi - mae_lo:.3f})")
    print(f"  AUC across the grid: {auc_lo:.3f} to {auc_hi:.3f}  (spread {auc_hi - auc_lo:.3f})")
    print(f"  shipped config ranks {mae_rank}/{len(frame)} by MAE, "
          f"{auc_rank}/{len(frame)} by AUC")
    print(f"\n  For reference, the leakage effect on the same pipeline is ~6.8 MAE and")
    print(f"  ~0.37 AUC (see `python run.py leakage-demo`). If the spread above is small")
    print(f"  next to that, the unrecorded hyperparameter search cannot account for the")
    print(f"  reported result, whatever it was.")

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "sensitivity.csv")
    frame.to_csv(path, index=False)
    print(f"  -> saved {path}")
    return frame


def main():
    if len(sys.argv) < 2:
        raise SystemExit(USAGE)
    cmd = sys.argv[1]
    outdir = "results"
    if "--out" in sys.argv:
        outdir = sys.argv[sys.argv.index("--out") + 1]
    if cmd == "provenance":
        print_provenance()
    elif cmd == "sensitivity":
        run_sensitivity(outdir)
    elif cmd == "nested":
        run_nested(outdir)
    else:
        raise SystemExit(USAGE)


if __name__ == "__main__":
    main()
