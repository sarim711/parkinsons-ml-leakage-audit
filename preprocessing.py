"""
Preprocessing for the UCI Parkinson's Telemonitoring severity task.

Provides:
  - data loading and column normalization
  - severity label definitions (3-class and binary)
  - the 39-feature engineering block
  - patient-disjoint cross-validation folds
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
# Committed dataset location, relative to the repository root.
DATA_PATH = os.path.join("data", "raw", "parkinsons_updrs.data")

# Patients with a single recording have an undefined sample standard deviation
# (pandas groupby .std() uses ddof=1, so a one-row group yields NaN). Only the
# `*_std` aggregate columns can be affected; mean/min/max/median are always
# defined. Those NaNs are filled with this constant so the design matrix stays
# finite for estimators that reject NaN.
SINGLE_RECORDING_STD_FILL = 0.0

# raw UCI column name -> clean name
RAW_MAP = {
    "Jitter(%)": "jitter", "Jitter(Abs)": "jitter_abs", "Jitter:RAP": "jitter_rap",
    "Jitter:PPQ5": "jitter_ppq5", "Jitter:DDP": "jitter_ddp",
    "Shimmer": "shimmer", "Shimmer(dB)": "shimmer_db", "Shimmer:APQ3": "shimmer_apq3",
    "Shimmer:APQ5": "shimmer_apq5", "Shimmer:APQ11": "shimmer_apq11", "Shimmer:DDA": "shimmer_dda",
    "NHR": "nhr", "HNR": "hnr", "RPDE": "rpde", "DFA": "dfa", "PPE": "ppe",
}
RAW16 = list(RAW_MAP.values())

# 11 log-transformed features: 5 jitter variants + 6 shimmer variants
LOG_FEATS = ["jitter", "jitter_abs", "jitter_rap", "jitter_ppq5", "jitter_ddp",
             "shimmer", "shimmer_db", "shimmer_apq3", "shimmer_apq5",
             "shimmer_apq11", "shimmer_dda"]


def resolve_data_path(path=DATA_PATH):
    """Return `path` if it exists, else resolve it against the repository root.

    Lets `run.py`, `tune.py` and pytest load the dataset regardless of the
    working directory they are invoked from. Does not affect any modelling
    decision.
    """
    if os.path.exists(path):
        return path
    root = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(root, path),                    # <root>/data/raw/<file>
                      os.path.join(root, os.path.basename(path))):  # <root>/<file>
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"could not find {path!r} relative to the working directory or {root!r}; "
        "download it from https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring"
    )


def load_raw(path=DATA_PATH):
    df = pd.read_csv(resolve_data_path(path))
    df = df.rename(columns={"subject#": "subject_id"})
    df = df.rename(columns=RAW_MAP)
    return df


def build_features(df):
    """Return the 39-feature design matrix as a DataFrame."""
    X = pd.DataFrame(index=df.index)

    # demographics (3)
    X["age"] = df["age"]
    X["sex"] = df["sex"]
    X["test_time"] = df["test_time"]

    # raw voice (16)
    for c in RAW16:
        X[c] = df[c]

    # log-transformed jitter/shimmer variants (11)
    for c in LOG_FEATS:
        X["log_" + c] = np.log1p(df[c])

    # ratios (4)
    X["shimmer_over_jitter"] = df["shimmer"] / (df["jitter"] + 1e-9)
    X["nhr_over_hnr"] = df["nhr"] / (df["hnr"] + 1e-9)
    X["ppe_dfa"] = df["ppe"] * df["dfa"]
    X["rpde_ppe"] = df["rpde"] * df["ppe"]

    # age interactions (5)
    X["age_sq"] = df["age"] ** 2
    X["age_ppe"] = df["age"] * df["ppe"]
    X["age_hnr"] = df["age"] * df["hnr"]
    X["age_dfa"] = df["age"] * df["dfa"]
    X["age_shimmer"] = df["age"] * df["shimmer"]

    return X


def patient_agg_features(df, groups):
    """Per-patient mean/std/min/max/median of the 16 raw voice features (80 columns).

    Computed across each patient's own recordings (no label use). Appended to the
    39-feature base gives 119 features for the patient-aggregation experiments.

    NOTE: these aggregates are a per-patient fingerprint. Under patient-disjoint
    evaluation they do not leak labels, but they do not transfer to unseen
    patients either -- see the aggregation result in the README.
    """
    out = pd.DataFrame(index=df.index)
    tmp = df.copy()
    tmp["__g"] = groups
    for f in RAW16:
        g = tmp.groupby("__g")[f]
        for stat in ["mean", "std", "min", "max", "median"]:
            out[f"{f}_{stat}"] = tmp["__g"].map(getattr(g, stat)()).values
    # Only reachable for patients with a single recording; see the constant's docstring.
    return out.fillna(SINGLE_RECORDING_STD_FILL)


def severity_3class(v):
    """0 = Mild (<20), 1 = Moderate ([20,30)), 2 = Severe (>=30)."""
    return 0 if v < 20 else (1 if v < 30 else 2)


def get_xy(df):
    """Return (X, y_reg, y3, y2, groups)."""
    X = build_features(df)
    y_reg = df["motor_UPDRS"].values
    y3 = np.array([severity_3class(v) for v in y_reg])
    y2 = (y_reg >= 20).astype(int)
    groups = df["subject_id"].values
    return X, y_reg, y3, y2, groups


def _patient_majority(y_cls, groups):
    """Per-patient majority class, broadcast back to every row (stratification target)."""
    tmp = pd.DataFrame({"g": groups, "y": y_cls})
    maj = tmp.groupby("g")["y"].agg(lambda s: s.value_counts().idxmax())
    return tmp["g"].map(maj).values


def make_folds(df, y_cls, groups, n_splits=5):
    """Patient-disjoint StratifiedGroupKFold.

    Stratifies on the per-patient majority of `y_cls`: pass the binary label (y2)
    for the binary / regression tracks and the 3-class label (y3) for the 3-class tracks.
    shuffle=True with a fixed seed so folds are reproducible.
    """
    strat = _patient_majority(y_cls, groups)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(sgkf.split(np.zeros(len(df)), strat, groups))
