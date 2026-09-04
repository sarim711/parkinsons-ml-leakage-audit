"""
Tests for the two invariants the repository's central claim rests on:

  1. cross-validation folds are patient-disjoint
  2. the Yeo-Johnson transform is fitted on training rows only

If either regresses, every reported number becomes an overestimate, so these are
asserted rather than assumed.
"""
import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PowerTransformer

import preprocessing as P
import evaluate as E


# --------------------------------------------------------------- 1. fold hygiene
@pytest.mark.parametrize("target", ["binary", "3class"])
def test_folds_are_patient_disjoint(data, target):
    """No subject id may appear in both sides of any fold, for either target."""
    df, X, y_reg, y3, y2, groups = data
    y_cls = y2 if target == "binary" else y3
    folds = P.make_folds(df, y_cls, groups)

    assert len(folds) == 5
    for k, (tr, te) in enumerate(folds):
        overlap = set(groups[tr]) & set(groups[te])
        assert overlap == set(), (
            f"{target} fold {k}: {len(overlap)} subject(s) in both train and test: "
            f"{sorted(overlap)[:5]}"
        )


@pytest.mark.parametrize("target", ["binary", "3class"])
def test_folds_partition_every_row_exactly_once(data, target):
    """Every recording is tested exactly once, so out-of-fold pooling is valid."""
    df, X, y_reg, y3, y2, groups = data
    y_cls = y2 if target == "binary" else y3
    folds = P.make_folds(df, y_cls, groups)

    test_idx = np.concatenate([te for _, te in folds])
    assert len(test_idx) == len(df)
    assert len(np.unique(test_idx)) == len(df)


@pytest.mark.parametrize("target", ["binary", "3class"])
def test_every_subject_is_held_out_exactly_once(data, target):
    df, X, y_reg, y3, y2, groups = data
    y_cls = y2 if target == "binary" else y3
    folds = P.make_folds(df, y_cls, groups)

    held_out = [s for _, te in folds for s in np.unique(groups[te])]
    assert sorted(held_out) == sorted(np.unique(groups))


def test_evaluate_rejects_a_row_level_split(data):
    """The guard in evaluate.py must fire on the leaky protocol.

    This is the splitter run.py's `leakage-demo` uses with assert_disjoint=False;
    with the guard active it has to raise.
    """
    df, X, y_reg, y3, y2, groups = data
    row_folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=P.SEED)
                     .split(np.zeros(len(df)), y2))

    with pytest.raises(AssertionError, match="patient leakage"):
        E.evaluate_regression_threshold(
            X, y_reg, groups, row_folds,
            lambda: ExtraTreesRegressor(n_estimators=2, random_state=P.SEED),
            yeo_johnson=True)


# ------------------------------------------------- 2. transform fitted on train only
def _split_for_transform(data):
    df, X, y_reg, y3, y2, groups = data
    tr, te = P.make_folds(df, y2, groups)[0]
    return X.values[tr], X.values[te]


def test_yeo_johnson_train_output_ignores_test_rows(data):
    """Perturbing the test rows must not move the transformed training rows.

    If the transform were fitted on the concatenation, changing Xte would shift
    the fitted lambdas and therefore the training output too.
    """
    Xtr, Xte = _split_for_transform(data)

    a1, b1 = E._fold_preprocess(Xtr, Xte, yeo_johnson=True)
    a2, b2 = E._fold_preprocess(Xtr, Xte * 1000.0, yeo_johnson=True)

    np.testing.assert_allclose(a1, a2, rtol=0, atol=0,
                               err_msg="training output changed when only test rows changed "
                                       "-- the transform saw test data")
    assert not np.allclose(b1, b2), "test perturbation had no effect; the test is not exercising anything"


def test_yeo_johnson_lambdas_match_a_train_only_fit(data):
    """The in-fold transform must equal PowerTransformer fitted on train alone."""
    Xtr, Xte = _split_for_transform(data)

    reference = PowerTransformer(method="yeo-johnson").fit(Xtr)
    a, b = E._fold_preprocess(Xtr, Xte, yeo_johnson=True)

    np.testing.assert_allclose(a, reference.transform(Xtr), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(b, reference.transform(Xte), rtol=1e-10, atol=1e-10)

    pooled = PowerTransformer(method="yeo-johnson").fit(np.vstack([Xtr, Xte]))
    assert not np.allclose(reference.lambdas_, pooled.lambdas_), (
        "train-only and pooled fits coincide; this dataset cannot distinguish them")


def test_power_transformer_standardizes_by_default():
    """Ridge / SVR rely on this: they are never explicitly scaled.

    run.py passes them yeo_johnson=True and no scale=True, so their
    standardization comes entirely from PowerTransformer's default.
    """
    assert PowerTransformer(method="yeo-johnson").get_params()["standardize"] is True

    rng = np.random.default_rng(0)
    out = PowerTransformer(method="yeo-johnson").fit_transform(rng.gamma(2.0, size=(500, 3)))
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-6)


def test_scale_and_pca_are_also_train_only(data):
    """Same guarantee for the other two optional in-fold transforms."""
    Xtr, Xte = _split_for_transform(data)

    a1, _ = E._fold_preprocess(Xtr, Xte, scale=True, pca=20)
    a2, _ = E._fold_preprocess(Xtr, Xte * 1000.0, scale=True, pca=20)
    np.testing.assert_allclose(a1, a2, rtol=0, atol=0)
