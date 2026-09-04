"""
Run the experiments and save results.

Usage:
    python run.py <section> [--out results]

  section: 3class | 2class | literature | regression | regression-final
           | leakage-demo | all

Examples:
    python run.py regression-final     # winning pipeline + per-fold table + OOF confusion
    python run.py leakage-demo         # row-level vs patient-disjoint splitting
    python run.py 2class
    python run.py all

Results are printed and written to CSV/JSON under the chosen --out directory.
NOTE: tree ensembles (500/700 trees) are CPU-heavy; on a single core each takes a few
minutes. Run sections individually if needed.
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import preprocessing as P
import models as M
import evaluate as E

USAGE = ("usage: python run.py "
         "{3class|2class|literature|regression|regression-final|leakage-demo|all} "
         "[--out results]")


# --------------------------------------------------------------------- sections
def run_3class(df, X, y3, groups):
    folds = P.make_folds(df, y3, groups)              # 3-class stratification
    rows = []
    # (name, factory, needs_y, extra kwargs)
    specs = [
        ("LogisticRegression", lambda: M.clf_logreg(),       False, dict(scale=True)),
        ("RandomForest",       lambda: M.clf_random_forest(), False, {}),
        ("XGBoost",            lambda: M.clf_xgboost(),       False, dict(balanced_sample_weight=True)),
        ("LightGBM",           lambda: M.clf_lightgbm(),      False, {}),
        ("ExtraTrees",         lambda: M.clf_extra_trees(),   False, {}),
    ]
    for name, mk, needs_y, kw in specs:
        r = E.evaluate_classifier(X, y3, groups, folds, mk, n_classes=3, needs_y=needs_y, **kw)
        r["model"] = name; rows.append(r)
        print(f"[3class] {name:20s} acc={r['accuracy']:.3f} f1={r['macro_f1']:.3f} "
              f"bacc={r['balanced_acc']:.3f} f1_sev={r['f1_severe']:.3f}")
    return pd.DataFrame(rows)


def run_2class(df, X, y2, groups):
    folds = P.make_folds(df, y2, groups)              # binary stratification
    rows = []
    specs = [
        ("LogisticRegression", lambda: M.clf_logreg(),        False, dict(scale=True)),
        ("RandomForest",       lambda: M.clf_random_forest(),  False, {}),
        ("ExtraTrees",         lambda: M.clf_extra_trees(),    False, {}),
        ("XGBoost",            lambda ytr: M.clf_xgboost(ytr, binary=True), True, {}),
        ("LightGBM",           lambda: M.clf_lightgbm(),       False, {}),
    ]
    for name, mk, needs_y, kw in specs:
        r = E.evaluate_classifier(X, y2, groups, folds, mk, n_classes=2, needs_y=needs_y, **kw)
        r["model"] = name; rows.append(r)
        print(f"[2class] {name:20s} acc={r['accuracy']:.3f} f1={r['macro_f1']:.3f} "
              f"bacc={r['balanced_acc']:.3f} auc={r['auc']:.3f}")
    return pd.DataFrame(rows)


def run_literature(df, X, y2, groups):
    """Yeo-Johnson / PCA / patient-aggregated-feature variants (binary task)."""
    folds = P.make_folds(df, y2, groups)
    Xbase = X
    Xpat = pd.concat([X.reset_index(drop=True),
                      P.patient_agg_features(df, groups).reset_index(drop=True)], axis=1)
    rows = []
    # (name, design matrix, factory, needs_y, extra kwargs)
    specs = [
        ("A_Yeo_ExtraTrees",        Xbase, lambda: M.clf_extra_trees(), False, dict(yeo_johnson=True)),
        ("B_Yeo_PCA_ExtraTrees",    Xbase, lambda: M.clf_extra_trees(), False, dict(yeo_johnson=True, pca=20)),
        ("C1_PatAgg_LightGBM",      Xpat,  lambda: M.clf_lightgbm(),    False, {}),
        ("C2_PatAgg_ExtraTrees",    Xpat,  lambda: M.clf_extra_trees(), False, {}),
        ("C3_PatAgg_XGBoost",       Xpat,  lambda ytr: M.clf_xgboost(ytr, binary=True), True, {}),
        ("D_PatAgg_Yeo_ExtraTrees", Xpat,  lambda: M.clf_extra_trees(), False, dict(yeo_johnson=True)),
        ("E_PatAgg_Yeo_LightGBM",   Xpat,  lambda: M.clf_lightgbm(),    False, dict(yeo_johnson=True)),
    ]
    for name, Xd, mk, needs_y, kw in specs:
        r = E.evaluate_classifier(Xd, y2, groups, folds, mk, n_classes=2, needs_y=needs_y, **kw)
        r["model"] = name; rows.append(r)
        print(f"[lit] {name:28s} acc={r['accuracy']:.3f} f1={r['macro_f1']:.3f} "
              f"bacc={r['balanced_acc']:.3f} auc={r['auc']:.3f}")
    return pd.DataFrame(rows)


def run_regression(df, X, y_reg, y2, groups):
    folds = P.make_folds(df, y2, groups)
    rows = []
    # (name, factory, apply Yeo-Johnson)
    specs = [
        ("Ridge",                M.reg_ridge,             True),
        ("SVR_RBF",              M.reg_svr,               True),
        ("ExtraTrees",           M.reg_extra_trees,       False),
        ("ExtraTrees_YeoJohnson", M.reg_extra_trees,      True),
        ("RandomForest",         M.reg_random_forest,     False),
        ("XGBoost",              M.reg_xgboost,           False),
        ("LightGBM",             M.reg_lightgbm,          False),
        ("GradientBoosting",     M.reg_gradient_boosting, False),
    ]
    for name, mk, yeo in specs:
        r = E.evaluate_regression_threshold(X, y_reg, groups, folds, mk, yeo_johnson=yeo)
        r["model"] = name; rows.append(r)
        print(f"[reg] {name:22s} mae={r['mae']:.2f} acc={r['accuracy']:.3f} "
              f"f1={r['macro_f1']:.3f} bacc={r['balanced_acc']:.3f} auc={r['auc']:.3f}")
    return pd.DataFrame(rows)


def run_regression_final(df, X, y_reg, y2, groups, outdir):
    """Winning pipeline: ExtraTreesRegressor + Yeo-Johnson, threshold 20, with per-fold detail."""
    folds = P.make_folds(df, y2, groups)
    r = E.evaluate_regression_threshold(X, y_reg, groups, folds, M.reg_extra_trees,
                                        yeo_johnson=True, return_folds=True)
    print("\n=== Winning pipeline: ExtraTreesRegressor + Yeo-Johnson -> threshold 20 ===")
    print(f"MAE={r['mae']:.2f}±{r['mae_std']:.2f}  AUC={r['auc']:.3f}±{r['auc_std']:.2f}  "
          f"Acc={r['accuracy']:.3f}  MacroF1={r['macro_f1']:.3f}  BalAcc={r['balanced_acc']:.3f}")
    print("per-fold:")
    for f in r["per_fold"]:
        print(f"  fold {f['fold']}: MAE={f['mae']:.2f} AUC={f['auc']:.3f} "
              f"Acc={f['accuracy']:.3f} MacroF1={f['macro_f1']:.3f} BalAcc={f['balanced_acc']:.3f}")
    print("pooled OOF confusion [[true_mild],[true_notmild]] x [pred_mild, pred_notmild]:")
    print("  ", r["oof_confusion"])
    print("pooled OOF:", {k: round(v, 3) for k, v in r["oof_pooled"].items()})
    with open(os.path.join(outdir, "regression_final.json"), "w") as fh:
        json.dump(r, fh, indent=2)
    pd.DataFrame(r["per_fold"]).to_csv(os.path.join(outdir, "regression_final_folds.csv"), index=False)
    return r


def run_leakage_demo(df, X, y_reg, y2, groups, outdir):
    """The repository's central claim, measured directly.

    Identical features, model (ExtraTreesRegressor), transform (Yeo-Johnson),
    threshold (20) and seed. The only thing that changes is the splitter:

      row-level      StratifiedKFold over recordings -- a patient's recordings are
                     scattered across train and test, so the model can recognise
                     the patient rather than read the voice signal.
      patient-disjoint StratifiedGroupKFold grouped by subject_id -- every test
                     patient is unseen during training.

    The gap between the two is the size of the leakage artefact.
    """
    row_folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=P.SEED)
                     .split(np.zeros(len(df)), y2))
    grp_folds = P.make_folds(df, y2, groups)

    shared = [len(np.intersect1d(groups[tr], groups[te])) for tr, te in row_folds]
    print("\n=== Leakage demonstration: row-level vs patient-disjoint splitting ===")
    print(f"subjects present in BOTH train and test, per fold: row-level={shared}  "
          f"patient-disjoint={[len(np.intersect1d(groups[tr], groups[te])) for tr, te in grp_folds]}")

    # assert_disjoint=False: the row-level protocol violates the invariant on purpose.
    leaky = E.evaluate_regression_threshold(X, y_reg, groups, row_folds, M.reg_extra_trees,
                                            yeo_johnson=True, assert_disjoint=False)
    honest = E.evaluate_regression_threshold(X, y_reg, groups, grp_folds, M.reg_extra_trees,
                                             yeo_johnson=True)

    for label, r in (("row-level (leaky)", leaky), ("patient-disjoint", honest)):
        print(f"  {label:20s} MAE={r['mae']:.2f}±{r['mae_std']:.2f}  "
              f"AUC={r['auc']:.3f}±{r['auc_std']:.3f}  Acc={r['accuracy']:.3f}  "
              f"MacroF1={r['macro_f1']:.3f}  BalAcc={r['balanced_acc']:.3f}")
    print(f"  {'delta (leaky - honest)':20s} MAE={leaky['mae'] - honest['mae']:+.2f}  "
          f"AUC={leaky['auc'] - honest['auc']:+.3f}  "
          f"Acc={leaky['accuracy'] - honest['accuracy']:+.3f}  "
          f"MacroF1={leaky['macro_f1'] - honest['macro_f1']:+.3f}  "
          f"BalAcc={leaky['balanced_acc'] - honest['balanced_acc']:+.3f}")
    print("  A lower MAE and higher AUC on the left column is the leakage artefact,")
    print("  not a better model: the two columns differ only in the splitter.")

    cols = ["mae", "mae_std", "accuracy", "macro_f1", "balanced_acc", "auc", "auc_std"]
    frame = pd.DataFrame([{"protocol": "row_level_stratified_kfold", **{c: leaky[c] for c in cols}},
                          {"protocol": "patient_disjoint_stratified_group_kfold",
                           **{c: honest[c] for c in cols}}])
    path = os.path.join(outdir, "leakage_comparison.csv")
    frame.to_csv(path, index=False)
    print(f"  -> saved {path}")
    return frame


# --------------------------------------------------------------------- main
def main():
    section = sys.argv[1] if len(sys.argv) > 1 else "all"
    valid = {"3class", "2class", "literature", "regression",
             "regression-final", "leakage-demo", "all"}
    if section not in valid:
        raise SystemExit(USAGE)
    outdir = "results"
    if "--out" in sys.argv:
        outdir = sys.argv[sys.argv.index("--out") + 1]
    os.makedirs(outdir, exist_ok=True)

    df = P.load_raw()
    X, y_reg, y3, y2, groups = P.get_xy(df)
    print(f"loaded {len(df)} recordings, {df['subject_id'].nunique()} patients, "
          f"{X.shape[1]} features")

    def save(name, frame):
        path = os.path.join(outdir, f"{name}.csv")
        frame.to_csv(path, index=False)
        print(f"  -> saved {path}")

    if section in ("3class", "all"):
        save("results_3class", run_3class(df, X, y3, groups))
    if section in ("2class", "all"):
        save("results_2class", run_2class(df, X, y2, groups))
    if section in ("literature", "all"):
        save("results_literature", run_literature(df, X, y2, groups))
    if section in ("regression", "all"):
        save("results_regression", run_regression(df, X, y_reg, y2, groups))
    if section in ("regression-final", "all"):
        run_regression_final(df, X, y_reg, y2, groups, outdir)
    if section in ("leakage-demo", "all"):
        run_leakage_demo(df, X, y_reg, y2, groups, outdir)


if __name__ == "__main__":
    main()
