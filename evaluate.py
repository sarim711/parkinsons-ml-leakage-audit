"""
Cross-validated evaluation routines.

Two evaluation modes:
  - classification: fit a classifier on the engineered features, score directly.
  - regression-then-threshold: fit a regressor on continuous motor_UPDRS,
    then threshold predictions at 20 to obtain the binary label.

Optional preprocessing inside each fold (fit on train, applied to test):
  - Yeo-Johnson power transform
  - PCA
All metrics are reported as mean and sample std (ddof=1) across folds.

Both entry points verify that no subject appears in both sides of a split. This
is the repository's central methodological claim, so it is asserted at run time
rather than left to convention. The `leakage-demo` section in run.py is the one
caller that deliberately opts out, via assert_disjoint=False.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             roc_auc_score, confusion_matrix)

THRESHOLD = 20.0


def _sd(x):
    return float(np.std(x, ddof=1))


def _check_disjoint(groups, tr, te, fold_idx):
    """Raise if any subject id appears in both the train and test side of a fold."""
    if groups is None:
        return
    overlap = np.intersect1d(np.asarray(groups)[tr], np.asarray(groups)[te])
    if overlap.size:
        raise AssertionError(
            f"patient leakage in fold {fold_idx}: {overlap.size} subject id(s) "
            f"appear in both train and test (e.g. {overlap[:5].tolist()})"
        )


def _fold_preprocess(Xtr, Xte, scale=False, yeo_johnson=False, pca=None):
    """Fit every transform on the training rows only, then apply to both sides.

    Xte is never passed to a .fit() call; this is what tests/test_folds.py checks.
    """
    if scale:
        sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    if yeo_johnson:
        # PowerTransformer defaults to standardize=True, so this also zero-means
        # and unit-scales the features (relevant for Ridge / SVR, which are
        # otherwise passed raw features).
        pt = PowerTransformer(method="yeo-johnson"); Xtr = pt.fit_transform(Xtr); Xte = pt.transform(Xte)
    if pca is not None:
        pc = PCA(n_components=pca, random_state=0); Xtr = pc.fit_transform(Xtr); Xte = pc.transform(Xte)
    return Xtr, Xte


def evaluate_classifier(X, y, groups, folds, make_model, n_classes,
                        scale=False, yeo_johnson=False, pca=None,
                        balanced_sample_weight=False, needs_y=False,
                        assert_disjoint=True):
    """Direct classification CV. Returns a dict of mean/std metrics.

    needs_y -- True when the model factory takes the training labels as its one
    argument (XGBoost's scale_pos_weight is computed from them). Declared
    explicitly by each spec in run.py rather than inferred from the signature.
    """
    Xv = X.values if hasattr(X, "values") else X
    acc, f1, bacc, auc, f1_sev = [], [], [], [], []
    for k, (tr, te) in enumerate(folds):
        if assert_disjoint:
            _check_disjoint(groups, tr, te, k)
        a, b = _fold_preprocess(Xv[tr], Xv[te], scale=scale, yeo_johnson=yeo_johnson, pca=pca)
        model = make_model(y[tr]) if needs_y else make_model()
        if balanced_sample_weight:
            model.fit(a, y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
        else:
            model.fit(a, y[tr])
        pred = model.predict(b)
        yte = y[te]
        acc.append(accuracy_score(yte, pred))
        f1.append(f1_score(yte, pred, average="macro"))
        bacc.append(balanced_accuracy_score(yte, pred))
        if n_classes == 2:
            prob = model.predict_proba(b)[:, 1]
            auc.append(roc_auc_score(yte, prob) if len(np.unique(yte)) > 1 else 0.5)
        else:
            f1_sev.append(f1_score(yte, pred, labels=[2], average="macro", zero_division=0))
    res = {"accuracy": np.mean(acc), "accuracy_std": _sd(acc),
           "macro_f1": np.mean(f1), "macro_f1_std": _sd(f1),
           "balanced_acc": np.mean(bacc), "balanced_acc_std": _sd(bacc)}
    if n_classes == 2:
        res["auc"] = np.mean(auc); res["auc_std"] = _sd(auc)
    else:
        res["f1_severe"] = np.mean(f1_sev)
    return res


def evaluate_regression_threshold(X, y_reg, groups, folds, make_model,
                                  yeo_johnson=False, return_folds=False,
                                  assert_disjoint=True):
    """Regression-then-threshold CV. Threshold predictions at 20 for binary metrics."""
    Xv = X.values if hasattr(X, "values") else X
    mae, acc, f1, bacc, auc = [], [], [], [], []
    oof_pred = np.full(len(y_reg), np.nan); oof_true = np.zeros(len(y_reg), int)
    per_fold = []
    for k, (tr, te) in enumerate(folds):
        if assert_disjoint:
            _check_disjoint(groups, tr, te, k)
        a, b = _fold_preprocess(Xv[tr], Xv[te], yeo_johnson=yeo_johnson)
        model = make_model(); model.fit(a, y_reg[tr]); pred = model.predict(b)
        m = float(np.mean(np.abs(pred - y_reg[te])))
        yt = (y_reg[te] >= THRESHOLD).astype(int)
        yh = (pred >= THRESHOLD).astype(int)
        a_auc = roc_auc_score(yt, pred) if len(np.unique(yt)) > 1 else 0.5
        mae.append(m); acc.append(accuracy_score(yt, yh)); f1.append(f1_score(yt, yh, average="macro"))
        bacc.append(balanced_accuracy_score(yt, yh)); auc.append(a_auc)
        oof_pred[te] = pred; oof_true[te] = yt
        per_fold.append({"fold": k, "mae": m, "auc": a_auc,
                         "accuracy": accuracy_score(yt, yh),
                         "macro_f1": f1_score(yt, yh, average="macro"),
                         "balanced_acc": balanced_accuracy_score(yt, yh)})
    res = {"mae": np.mean(mae), "mae_std": _sd(mae),
           "accuracy": np.mean(acc), "accuracy_std": _sd(acc),
           "macro_f1": np.mean(f1), "macro_f1_std": _sd(f1),
           "balanced_acc": np.mean(bacc), "balanced_acc_std": _sd(bacc),
           "auc": np.mean(auc), "auc_std": _sd(auc)}
    if return_folds:
        yh = (oof_pred >= THRESHOLD).astype(int)
        res["per_fold"] = per_fold
        res["oof_confusion"] = confusion_matrix(oof_true, yh).tolist()
        res["oof_pooled"] = {"accuracy": accuracy_score(oof_true, yh),
                             "macro_f1": f1_score(oof_true, yh, average="macro"),
                             "balanced_acc": balanced_accuracy_score(oof_true, yh),
                             "auc": roc_auc_score(oof_true, oof_pred)}
    return res
