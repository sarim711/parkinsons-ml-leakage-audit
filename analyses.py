"""
Supporting analyses reported in the paper, beyond the model comparisons in run.py.

Usage:
    python analyses.py <section> [--out results]

  section: describe | baselines | rowlevel | scoreclass | features | patients
           | calibration | smote | all

  describe     the cohort description, Table 1 and the correlations with motor UPDRS
  baselines    patient-average baseline (no voice), overall-average baseline,
               and the final pipeline's MAE / RMSE / R^2 / AUC per fold
  rowlevel     the deliberately leaky split by recording: the final pipeline with
               all features, with no voice features, and with voice only; and the
               patient-summary features under the same split
  scoreclass   regression-then-cut-off against the direct classifier, over the
               reporting folds and 10 further fold assignments
  features     final pipeline on feature groups (all, demographics, voice, ...)
               over the reporting folds and 10 further fold assignments, with
               corrected resampled t intervals for the paired differences
  patients     patients-versus-recordings experiment
  calibration  calibration with a new patient's own recordings: random
               recordings and the patient's first recordings in time
  smote        SMOTE / Borderline-SMOTE on the three-class task

Every fold is patient-disjoint, except in `rowlevel`, which splits by recording on
purpose to measure what leakage gives.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

import preprocessing as P
import models as M
import evaluate as E

SECTIONS = ("describe", "baselines", "rowlevel", "scoreclass", "features",
            "patients", "calibration", "smote")
USAGE = f"usage: python analyses.py {{{'|'.join(SECTIONS)}|all}} [--out results]"

FOLD_SEEDS = list(range(10))          # extra fold assignments for the repeated analysis
DEMOGRAPHIC = ["age", "sex", "test_time"]


def _auc(y_reg, pred):
    yt = (y_reg >= E.THRESHOLD).astype(int)
    return roc_auc_score(yt, pred) if len(np.unique(yt)) > 1 else 0.5


def _final_pipeline(Xtr, ytr, Xte, n_estimators=700):
    """Yeo-Johnson fitted on training rows -> ExtraTreesRegressor, as in models.py."""
    a, b = E._fold_preprocess(Xtr, Xte, yeo_johnson=True)
    m = ExtraTreesRegressor(n_estimators, min_samples_leaf=2, random_state=P.SEED, n_jobs=-1)
    m.fit(a, ytr)
    return m.predict(b)


def _folds_for_seed(y2, groups, seed):
    strat = P._patient_majority(y2, groups)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(groups)), strat, groups))


def _check(groups, folds):
    for k, (tr, te) in enumerate(folds):
        E._check_disjoint(groups, tr, te, k)


# --------------------------------------------------------------------- describe
def run_describe(df, X, y_reg, y3, y2, groups, outdir):
    """Cohort description, the class counts of Table 1, and Pearson r with motor UPDRS."""
    rows = []

    def add(name, value):
        rows.append({"name": name, "value": float(value)})

    per_patient = df.groupby("subject_id")
    n, first = per_patient.size(), per_patient.first()
    add("recordings", len(df)); add("patients", len(n))
    add("recordings_per_patient_min", n.min()); add("recordings_per_patient_max", n.max())
    add("recordings_per_patient_mean", n.mean())
    add("age_min", first["age"].min()); add("age_max", first["age"].max())
    add("age_mean", first["age"].mean()); add("age_sd", first["age"].std(ddof=1))
    add("patients_male", (first["sex"] == 0).sum()); add("patients_female", (first["sex"] == 1).sum())
    m = df["motor_UPDRS"]
    add("motor_updrs_min", m.min()); add("motor_updrs_max", m.max()); add("motor_updrs_mean", m.mean())
    add("motor_updrs_sd", m.std(ddof=1)); add("motor_updrs_median", m.median())
    add("test_time_min", df["test_time"].min()); add("recordings_with_negative_test_time", (df["test_time"] < 0).sum())

    schemes = (("three_class", y3, ["mild", "moderate", "severe"]),
               ("binary", y2, ["mild", "moderate_to_severe"]))
    for scheme, y, names in schemes:
        majority = pd.Series(P._patient_majority(y, groups)).groupby(groups).first()
        for c, name in enumerate(names):
            add(f"{scheme}_{name}_recordings", (y == c).sum())
            add(f"{scheme}_{name}_patients_by_majority", (majority == c).sum())
        add(f"{scheme}_patients_in_more_than_one_class", (pd.Series(y).groupby(groups).nunique() > 1).sum())

    # Over all recordings; age, sex and test time plus the 16 raw measures, largest |r| first.
    corr = df[DEMOGRAPHIC + P.RAW16].corrwith(m)
    for c in corr.abs().sort_values(ascending=False).index:
        add(f"pearson_r_with_motor_updrs_{c}", corr[c])

    frame = pd.DataFrame(rows)
    for _, r in frame.iterrows():
        print(f"[describe] {r['name']:44s} {r['value']:.3f}")
    return frame


# --------------------------------------------------------------------- baselines
def patient_average_baseline(y_reg, groups):
    """Mean of all *other* recordings of the same patient (leave-one-out). No voice."""
    s = pd.Series(y_reg, dtype=float)
    sums = s.groupby(groups).transform("sum").values
    cnts = s.groupby(groups).transform("count").values
    return (sums - y_reg) / (cnts - 1)


def run_baselines(df, X, y_reg, y2, groups, outdir):
    folds = P.make_folds(df, y2, groups)
    _check(groups, folds)
    y = y_reg.astype(float)
    rows = []

    oracle = patient_average_baseline(y, groups)
    err = oracle - y
    rows.append({"name": "patient_average_baseline", "mae": np.abs(err).mean(),
                 "rmse": np.sqrt((err ** 2).mean()),
                 "r2": 1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()})

    err = y.mean() - y
    rows.append({"name": "overall_average_global_mean", "mae": np.abs(err).mean(),
                 "rmse": np.sqrt((err ** 2).mean()), "r2": 0.0})

    mae, rmse = [], []
    for tr, te in folds:   # the leakage-free version: the mean of the training patients only
        e = y[tr].mean() - y[te]
        mae.append(np.abs(e).mean()); rmse.append(np.sqrt((e ** 2).mean()))
    rows.append({"name": "overall_average_training_mean", "mae": np.mean(mae), "rmse": np.mean(rmse),
                 "mae_std": np.std(mae, ddof=1), "rmse_std": np.std(rmse, ddof=1)})

    mae, rmse, r2, auc = [], [], [], []
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        pred = _final_pipeline(X.values[tr], y[tr], X.values[te])
        e = pred - y[te]
        mae.append(np.abs(e).mean()); rmse.append(np.sqrt((e ** 2).mean()))
        r2.append(1 - (e ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
        auc.append(_auc(y[te], pred)); oof[te] = pred
    rows.append({"name": "final_pipeline", "mae": np.mean(mae), "rmse": np.mean(rmse),
                 "r2": np.mean(r2), "auc": np.mean(auc), "mae_std": np.std(mae, ddof=1),
                 "rmse_std": np.std(rmse, ddof=1), "auc_std": np.std(auc, ddof=1),
                 "auc_pooled_oof": roc_auc_score((y >= E.THRESHOLD).astype(int), oof)})

    within = pd.Series(y).groupby(groups).std(ddof=1).mean()
    between = pd.Series(y).groupby(groups).mean().std(ddof=1)
    rows.append({"name": "within_patient_sd", "value": within})
    rows.append({"name": "between_patient_sd", "value": between})

    frame = pd.DataFrame(rows)
    for _, r in frame.iterrows():
        vals = "  ".join(f"{c}={r[c]:.3f}" for c in frame.columns if c != "name" and pd.notna(r[c]))
        print(f"[baselines] {r['name']:32s} {vals}")
    return frame


# --------------------------------------------------------------------- rowlevel
def _r2(y_true, pred):
    return 1 - ((pred - y_true) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()


def run_row_level(df, X, y_reg, y2, groups, outdir):
    """What the split by recording gives, with and without voice features.

    The folds are the row-level StratifiedKFold of `run.py leakage-demo`: every
    patient is in both the training and the test side, on purpose. No fold here
    is patient-disjoint, so nothing in this section estimates performance on a
    new patient. It measures how much of the near-perfect score needs the voice.
    """
    y = y_reg.astype(float)
    row_folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=P.SEED)
                     .split(np.zeros(len(df)), y2))
    shared = [len(np.intersect1d(groups[tr], groups[te])) for tr, te in row_folds]
    print(f"[rowlevel] patients in BOTH train and test, per fold: {shared}")

    rows = []
    groups_map = feature_groups(X.columns)
    for name in ("all", "demographics", "voice"):      # the final pipeline (Yeo-Johnson + ExtraTrees)
        Xg = X[groups_map[name]].values
        mae, rmse, r2, auc = [], [], [], []
        for tr, te in row_folds:
            pred = _final_pipeline(Xg[tr], y[tr], Xg[te])
            e = pred - y[te]
            mae.append(np.abs(e).mean()); rmse.append(np.sqrt((e ** 2).mean()))
            r2.append(_r2(y[te], pred)); auc.append(_auc(y[te], pred))
        rows.append({"model": "final_pipeline", "features": name, "n_features": Xg.shape[1],
                     "mae": np.mean(mae), "rmse": np.mean(rmse), "r2": np.mean(r2), "auc": np.mean(auc)})

    # Patient-summary features, same estimator and no transform as the
    # ExtraTrees / ExtraTrees_PatAgg rows of `run.py regression`.
    Xpat = pd.concat([X.reset_index(drop=True),
                      P.patient_agg_features(df, groups).reset_index(drop=True)], axis=1)
    for name, Xd in (("all", X), ("all_plus_patient_summaries", Xpat)):
        mae, rmse, r2, auc = [], [], [], []
        for tr, te in row_folds:
            pred = M.reg_extra_trees().fit(Xd.values[tr], y[tr]).predict(Xd.values[te])
            e = pred - y[te]
            mae.append(np.abs(e).mean()); rmse.append(np.sqrt((e ** 2).mean()))
            r2.append(_r2(y[te], pred)); auc.append(_auc(y[te], pred))
        rows.append({"model": "extra_trees_no_transform", "features": name, "n_features": Xd.shape[1],
                     "mae": np.mean(mae), "rmse": np.mean(rmse), "r2": np.mean(r2), "auc": np.mean(auc)})

    frame = pd.DataFrame(rows)
    for _, r in frame.iterrows():
        print(f"[rowlevel] {r['model']:26s} {r['features']:28s} MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}  "
              f"R2={r['r2']:.4f}  AUC={r['auc']:.4f}")
    return frame


# --------------------------------------------------------------------- scoreclass
def run_score_vs_class(df, X, y_reg, y2, groups, outdir):
    """Regression-then-cut-off (the final pipeline) against the direct ExtraTrees classifier.

    Both are scored by AUC on the same folds: the reporting folds, and the ten
    further fold assignments used in `features`. The interval for the paired
    difference is the corrected resampled t interval over those 50 folds.
    """
    Xv, rows, ratios = X.values, [], []
    for seed in [P.SEED] + FOLD_SEEDS:
        folds = P.make_folds(df, y2, groups) if seed == P.SEED else _folds_for_seed(y2, groups, seed)
        _check(groups, folds)
        for k, (tr, te) in enumerate(folds):
            score = _final_pipeline(Xv[tr], y_reg[tr], Xv[te])
            clf = M.clf_extra_trees().fit(Xv[tr], y2[tr])
            prob = clf.predict_proba(Xv[te])[:, 1]
            rows.append({"fold_seed": seed, "fold": k, "auc_regression": _auc(y_reg[te], score),
                         "auc_classifier": roc_auc_score(y2[te], prob) if len(np.unique(y2[te])) > 1 else 0.5})
            ratios.append(len(te) / len(tr))
        sub = pd.DataFrame([r for r in rows if r["fold_seed"] == seed])
        print(f"[scoreclass] seed {seed:2d}  regression AUC={sub['auc_regression'].mean():.3f}  "
              f"classifier AUC={sub['auc_classifier'].mean():.3f}", flush=True)

    folds_frame = pd.DataFrame(rows)
    folds_frame.to_csv(os.path.join(outdir, "score_vs_class_folds.csv"), index=False)

    rep = folds_frame[folds_frame["fold_seed"] == P.SEED]
    extra = folds_frame[folds_frame["fold_seed"] != P.SEED]
    per_seed = extra.groupby("fold_seed")[["auc_regression", "auc_classifier"]].mean()
    mean, lo, hi = corrected_t_interval(extra["auc_regression"] - extra["auc_classifier"], float(np.mean(ratios)))
    frame = pd.DataFrame([
        {"comparison": "reporting folds", "auc_regression": rep["auc_regression"].mean(),
         "auc_classifier": rep["auc_classifier"].mean()},
        {"comparison": "10 further fold assignments", "auc_regression": per_seed["auc_regression"].mean(),
         "auc_classifier": per_seed["auc_classifier"].mean(),
         "assignments_regression_ahead": int((per_seed["auc_regression"] > per_seed["auc_classifier"]).sum()),
         "assignments": len(per_seed)},
        {"comparison": "regression minus classifier (auc, 50 folds)", "diff": mean, "ci_low": lo, "ci_high": hi},
    ])
    for _, r in frame.iterrows():
        vals = "  ".join(f"{c}={r[c]:.3f}" for c in frame.columns if c != "comparison" and pd.notna(r[c]))
        print(f"[scoreclass] {r['comparison']:44s} {vals}")
    return frame


# --------------------------------------------------------------------- features
def feature_groups(columns):
    age_terms = [c for c in columns if c.startswith("age_")]
    return {
        "all": list(columns),
        "without_test_time": [c for c in columns if c != "test_time"],
        "demographics": DEMOGRAPHIC,
        "age_sex": ["age", "sex"],
        "voice": [c for c in columns if c not in DEMOGRAPHIC + age_terms],
    }


def corrected_t_interval(diffs, test_train_ratio, level=0.95):
    """Nadeau-Bengio corrected resampled t interval for paired fold differences."""
    d = np.asarray(diffs, float)
    j = len(d)
    se = np.sqrt((1.0 / j + test_train_ratio) * d.var(ddof=1))
    t = stats.t.ppf(0.5 + level / 2, j - 1)
    return d.mean(), d.mean() - t * se, d.mean() + t * se


def run_features(df, X, y_reg, y2, groups, outdir):
    groups_map = feature_groups(X.columns)
    plan = [(P.SEED, name) for name in groups_map]                       # reporting folds
    plan += [(s, name) for s in FOLD_SEEDS for name in ("all", "demographics", "voice")]

    rows, ratios = [], []
    for seed, name in plan:
        folds = P.make_folds(df, y2, groups) if seed == P.SEED else _folds_for_seed(y2, groups, seed)
        _check(groups, folds)
        Xg = X[groups_map[name]].values
        for k, (tr, te) in enumerate(folds):
            pred = _final_pipeline(Xg[tr], y_reg[tr], Xg[te])
            e = pred - y_reg[te]
            rows.append({"fold_seed": seed, "fold": k, "group": name, "mae": np.abs(e).mean(),
                         "rmse": np.sqrt((e ** 2).mean()), "auc": _auc(y_reg[te], pred)})
            ratios.append(len(te) / len(tr))
        sub = pd.DataFrame([r for r in rows if r["fold_seed"] == seed and r["group"] == name])
        print(f"[features] seed {seed:2d} {name:18s} MAE={sub['mae'].mean():.2f}  "
              f"RMSE={sub['rmse'].mean():.2f}  AUC={sub['auc'].mean():.3f}", flush=True)

    folds_frame = pd.DataFrame(rows)
    folds_frame.to_csv(os.path.join(outdir, "feature_groups_folds.csv"), index=False)

    summary = []
    rep = folds_frame[folds_frame["fold_seed"] == P.SEED]
    for name, sub in rep.groupby("group", sort=False):
        summary.append({"comparison": f"{name} (reporting folds)", "mae": sub["mae"].mean(),
                        "rmse": sub["rmse"].mean(), "auc": sub["auc"].mean(),
                        "auc_std": sub["auc"].std(ddof=1)})
    extra = folds_frame[folds_frame["fold_seed"] != P.SEED]
    per_seed = extra[extra["group"] == "all"].groupby("fold_seed")[["mae", "auc"]].mean()
    summary.append({"comparison": "all, 10 further fold assignments",
                    "mae": per_seed["mae"].mean(), "mae_min": per_seed["mae"].min(),
                    "mae_max": per_seed["mae"].max(), "auc": per_seed["auc"].mean(),
                    "auc_min": per_seed["auc"].min(), "auc_max": per_seed["auc"].max()})
    ratio = float(np.mean(ratios))
    wide = extra.pivot_table(index=["fold_seed", "fold"], columns="group", values=["auc", "mae"])
    for other in ("demographics", "voice"):
        for metric in ("auc", "mae"):
            mean, lo, hi = corrected_t_interval(wide[(metric, other)] - wide[(metric, "all")], ratio)
            summary.append({"comparison": f"{other} minus all ({metric}, 50 folds)",
                            "diff": mean, "ci_low": lo, "ci_high": hi})
    frame = pd.DataFrame(summary)
    for _, r in frame.iterrows():
        vals = "  ".join(f"{c}={r[c]:.3f}" for c in frame.columns if c != "comparison" and pd.notna(r[c]))
        print(f"[features] {r['comparison']:42s} {vals}")
    return frame


# --------------------------------------------------------------------- patients
PATIENT_COUNTS = [8, 16, 24, 33]
PATIENT_DRAWS = 20
# Target training volumes for the fixed-volume experiment. At 2,000 the eight-patient
# arm cannot reach the target (no patient has more than 168 recordings, so it gets
# about 1,100). At 800 every arm can: 8 x 100, 16 x 50, 24 x 33, 33 x 24.
PATIENT_VOLUMES = [2000, 800]


def run_patients(df, X, y_reg, y2, groups, outdir):
    """Patients versus recordings, on all five reporting folds.

    1. Keep every training patient and cap the recordings per patient.
    2. Hold the training volume fixed and vary how many patients supply it:
       PATIENT_DRAWS random draws of patients per setting, at each volume in
       PATIENT_VOLUMES. The spread reported is the sample SD over draws.
    """
    folds = P.make_folds(df, y2, groups)
    _check(groups, folds)
    Xv, g = X.values, groups
    rows = []

    rng = np.random.default_rng(P.SEED)
    for cap in [10, 30, 60, 100, None]:
        auc, mae, vol = [], [], []
        for tr, te in folds:
            keep = []
            for pid in np.unique(g[tr]):
                idx = tr[g[tr] == pid]
                if cap and len(idx) > cap:
                    idx = rng.choice(idx, cap, replace=False)
                keep.append(idx)
            keep = np.concatenate(keep)
            pred = _final_pipeline(Xv[keep], y_reg[keep], Xv[te], n_estimators=300)
            auc.append(_auc(y_reg[te], pred)); mae.append(np.abs(pred - y_reg[te]).mean())
            vol.append(len(keep))
        rows.append({"experiment": "recordings_per_patient", "level": cap or "all",
                     "auc": np.mean(auc), "mae": np.mean(mae), "train_recordings": np.mean(vol)})

    for volume in PATIENT_VOLUMES:
        for n_pat in PATIENT_COUNTS:
            draw_auc, draw_mae, vol = [], [], []
            for draw in range(PATIENT_DRAWS):
                r = np.random.default_rng(100 + draw)
                auc, mae = [], []
                for tr, te in folds:
                    pats = r.choice(np.unique(g[tr]), min(n_pat, len(np.unique(g[tr]))), replace=False)
                    cap = max(1, volume // n_pat)
                    keep = []
                    for pid in pats:
                        idx = tr[g[tr] == pid]
                        if len(idx) > cap:
                            idx = r.choice(idx, cap, replace=False)
                        keep.append(idx)
                    keep = np.concatenate(keep)
                    pred = _final_pipeline(Xv[keep], y_reg[keep], Xv[te], n_estimators=300)
                    auc.append(_auc(y_reg[te], pred)); mae.append(np.abs(pred - y_reg[te]).mean())
                    vol.append(len(keep))
                draw_auc.append(np.mean(auc)); draw_mae.append(np.mean(mae))
            rows.append({"experiment": f"patients_volume_{volume}", "level": n_pat,
                         "auc": np.mean(draw_auc), "auc_sd_over_draws": np.std(draw_auc, ddof=1),
                         "mae": np.mean(draw_mae), "mae_sd_over_draws": np.std(draw_mae, ddof=1),
                         "train_recordings": np.mean(vol)})
            print(f"[patients] volume {volume:5d}  patients={n_pat:2d} done", flush=True)

    frame = pd.DataFrame(rows)
    for _, r in frame.iterrows():
        print(f"[patients] {r['experiment']:24s} {str(r['level']):>4s}  AUC={r['auc']:.3f}  "
              f"MAE={r['mae']:.2f}  train recordings={r['train_recordings']:.0f}")
    return frame


# --------------------------------------------------------------------- calibration
CAL_KS = [0, 1, 2, 3, 5, 10, 20]
CAL_REPEATS = 25


def run_calibration(df, X, y_reg, y2, groups, outdir):
    """Notebook cell 39 (random calibration recordings), plus the patient's first k in time.

    The offset is the mean (true - predicted) over the k calibration recordings;
    error is measured on the patient's remaining recordings. The model is never
    retrained and no test recording is used for training.
    """
    folds = P.make_folds(df, y2, groups)
    _check(groups, folds)
    y, g, tt = y_reg.astype(float), groups, df["test_time"].values
    rand, rand_mean = {k: [] for k in CAL_KS}, {k: [] for k in CAL_KS}
    first, first_mean = {k: [] for k in CAL_KS}, {k: [] for k in CAL_KS}
    rng = np.random.default_rng(P.SEED)
    for tr, te in folds:
        base = _final_pipeline(X.values[tr], y[tr], X.values[te])
        tg, ty, tt_te = g[te], y[te], tt[te]
        for pid in np.unique(tg):
            idx = np.where(tg == pid)[0]
            if len(idx) < 25:
                continue
            order = idx[np.argsort(tt_te[idx], kind="stable")]
            for k in CAL_KS:
                if k == 0:
                    e0 = np.abs(base[idx] - ty[idx]).mean()
                    rand[0].append(e0); first[0].append(e0)
                    continue
                cc, mm = [], []
                for _ in range(CAL_REPEATS):
                    cal = rng.choice(idx, k, replace=False); ev = np.setdiff1d(idx, cal)
                    off = np.mean(ty[cal] - base[cal])
                    cc.append(np.abs(base[ev] + off - ty[ev]).mean())
                    mm.append(np.abs(ty[cal].mean() - ty[ev]).mean())
                rand[k].append(np.mean(cc)); rand_mean[k].append(np.mean(mm))
                cal, ev = order[:k], order[k:]
                first[k].append(np.abs(base[ev] + np.mean(ty[cal] - base[cal]) - ty[ev]).mean())
                first_mean[k].append(np.abs(ty[cal].mean() - ty[ev]).mean())

    frame = pd.DataFrame([{"k": k,
                           "random_model_offset": np.mean(rand[k]),
                           "random_calibration_mean": np.mean(rand_mean[k]) if k else np.nan,
                           "first_k_model_offset": np.mean(first[k]),
                           "first_k_calibration_mean": np.mean(first_mean[k]) if k else np.nan}
                          for k in CAL_KS])
    print("[calibration] MAE averaged over test patients")
    print(frame.round(2).to_string(index=False))
    return frame


# --------------------------------------------------------------------- smote
def run_smote(df, X, y3, groups, outdir):
    """Oversampling inside each training fold, three-class task.

    SMOTE interpolates between nearest neighbours. On the raw features those
    neighbours would be decided by the few columns with the largest numerical
    range (age squared runs into the thousands, jitter is of the order of
    0.005), so the neighbours are found on features standardized on the training
    fold, and the oversampled set is then mapped back to the original units.
    The baseline and the test recordings are left untouched, so the three rows
    differ only in the synthetic recordings added to training.
    """
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE

    folds3 = P.make_folds(df, y3, groups)
    _check(groups, folds3)
    Xv, rows = X.values, []
    for mode in ["baseline", "smote", "borderline"]:
        f1, f1_sev = [], []
        for tr, te in folds3:
            Xtr, ytr = Xv[tr], y3[tr]
            if mode == "baseline":
                clf = RandomForestClassifier(400, min_samples_leaf=2, class_weight="balanced",
                                             random_state=P.SEED, n_jobs=-1)
            else:
                sm = SMOTE(random_state=P.SEED) if mode == "smote" else BorderlineSMOTE(random_state=P.SEED)
                sc = StandardScaler().fit(Xtr)
                Xs, ytr = sm.fit_resample(sc.transform(Xtr), ytr)
                Xtr = sc.inverse_transform(Xs)
                clf = RandomForestClassifier(400, min_samples_leaf=2, random_state=P.SEED, n_jobs=-1)
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xv[te])
            f1.append(f1_score(y3[te], pred, average="macro"))
            f1_sev.append(f1_score(y3[te], pred, labels=[2], average="macro", zero_division=0))
        rows.append({"mode": mode, "macro_f1": np.mean(f1), "f1_severe": np.mean(f1_sev)})
        print(f"[smote] {mode:10s} macroF1={rows[-1]['macro_f1']:.3f}  F1severe={rows[-1]['f1_severe']:.3f}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- main
def main():
    section = sys.argv[1] if len(sys.argv) > 1 else "all"
    if section not in SECTIONS + ("all",):
        raise SystemExit(USAGE)
    outdir = "results"
    if "--out" in sys.argv:
        outdir = sys.argv[sys.argv.index("--out") + 1]
    os.makedirs(outdir, exist_ok=True)

    df = P.load_raw()
    X, y_reg, y3, y2, groups = P.get_xy(df)
    print(f"loaded {len(df)} recordings, {df['subject_id'].nunique()} patients, {X.shape[1]} features")

    def save(name, frame):
        path = os.path.join(outdir, f"{name}.csv")
        frame.to_csv(path, index=False)
        print(f"  -> saved {path}")

    if section in ("describe", "all"):
        save("dataset_description", run_describe(df, X, y_reg, y3, y2, groups, outdir))
    if section in ("baselines", "all"):
        save("baselines", run_baselines(df, X, y_reg, y2, groups, outdir))
    if section in ("rowlevel", "all"):
        save("row_level", run_row_level(df, X, y_reg, y2, groups, outdir))
    if section in ("calibration", "all"):
        save("calibration", run_calibration(df, X, y_reg, y2, groups, outdir))
    if section in ("smote", "all"):
        save("smote", run_smote(df, X, y3, groups, outdir))
    if section in ("patients", "all"):
        save("patients_vs_recordings", run_patients(df, X, y_reg, y2, groups, outdir))
    if section in ("scoreclass", "all"):
        save("score_vs_class_summary", run_score_vs_class(df, X, y_reg, y2, groups, outdir))
    if section in ("features", "all"):
        save("feature_groups_summary", run_features(df, X, y_reg, y2, groups, outdir))


if __name__ == "__main__":
    main()
