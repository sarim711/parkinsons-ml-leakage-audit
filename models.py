"""
Model definitions for every experiment.

Each factory returns a fresh, unfitted estimator with the hyper-parameters used in
the study. Class balancing is applied via class_weight / scale_pos_weight / sample_weight
depending on what each library supports (see evaluate.py for how sample weights are passed).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              ExtraTreesRegressor, RandomForestRegressor,
                              GradientBoostingRegressor)
import xgboost as xgb
import lightgbm as lgb
from sklearn.utils.class_weight import compute_class_weight

from preprocessing import SEED


# ---------------------------------------------------------------- classifiers
def clf_logreg():
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)


def clf_random_forest():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                  class_weight="balanced", random_state=SEED, n_jobs=-1)


def clf_extra_trees():
    return ExtraTreesClassifier(n_estimators=700, class_weight="balanced",
                                random_state=SEED, n_jobs=-1)


def clf_xgboost(y_train=None, binary=False):
    spw = None
    if binary and y_train is not None:
        cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
        spw = cw[1] / cw[0]
    return xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                             colsample_bytree=0.85, scale_pos_weight=spw,
                             random_state=SEED, n_jobs=-1)


def clf_lightgbm():
    return lgb.LGBMClassifier(n_estimators=800, num_leaves=63, learning_rate=0.04,
                              class_weight="balanced", random_state=SEED,
                              n_jobs=-1, verbose=-1)


# ---------------------------------------------------------------- regressors
def reg_ridge():
    # No random_state: Ridge's default solver ("auto" -> cholesky/svd for this
    # dense, well-conditioned problem) is deterministic. random_state only has
    # an effect for the "sag"/"saga" solvers, so it was a no-op here.
    return Ridge(alpha=1.0)


def reg_svr():
    return SVR(kernel="rbf", C=1.0, epsilon=1.0)


def reg_extra_trees():
    return ExtraTreesRegressor(n_estimators=700, min_samples_leaf=2,
                               random_state=SEED, n_jobs=-1)


def reg_random_forest():
    return RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)


def reg_xgboost():
    return xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
                            subsample=0.85, colsample_bytree=0.85,
                            random_state=SEED, n_jobs=-1)


def reg_lightgbm():
    return lgb.LGBMRegressor(n_estimators=800, num_leaves=63, learning_rate=0.04,
                             random_state=SEED, n_jobs=-1, verbose=-1)


def reg_gradient_boosting():
    return GradientBoostingRegressor(n_estimators=400, random_state=SEED)
