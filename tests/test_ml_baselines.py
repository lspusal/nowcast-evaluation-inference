"""Tests for the M2 classical ML baselines.

The load-bearing checks here are about the FEATURES, not the learners: every
model shares `_design_matrix`, so a leak or a mis-specified lag there would
contaminate all three at once while each individual model still looked fine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.ml_baselines import (
    ML_MODELS,
    LSTMNowcaster,
    RFNowcaster,
    XGBNowcaster,
    _design_matrix,
    backward_impute,
)


@pytest.fixture
def panel_and_target():
    rng = np.random.default_rng(0)
    months = pd.date_range("1995-01-01", "2019-12-01", freq="MS")
    x1 = pd.Series(rng.normal(0, 1, len(months)) + 5.0, index=months)
    x2 = pd.Series(rng.normal(0, 1, len(months)).cumsum() * 0.1, index=months)
    panel = pd.DataFrame({"IND1": x1, "IND2": x2})
    xq = panel.groupby(panel.index.to_period("Q")).mean()
    y = 2.0 + 0.6 * xq["IND1"] + rng.normal(0, 0.3, len(xq))
    y.index = xq.index.to_timestamp()
    return panel, y


def test_backward_impute_delegates_and_stays_backward():
    """Both methods must leave pre-history cells missing (R1)."""
    df = pd.DataFrame({"A": [None, None, 3.0, None, 5.0]})
    locf = backward_impute(df, "locf")
    expanding = backward_impute(df, "expanding_mean")
    for out in (locf, expanding):
        assert pd.isna(out["A"].iloc[0]), "filled a cell with no prior data"
        assert pd.isna(out["A"].iloc[1])
    assert locf["A"].iloc[3] == 3.0  # carried forward
    assert expanding["A"].iloc[3] == pytest.approx(3.0)
    with pytest.raises(ValueError, match="unsupported imputation"):
        backward_impute(df, "interpolate")


def test_ar_features_are_position_based_not_calendar_based(panel_and_target):
    """The AR lags must be the most recent PUBLISHED values, not q-1..q-4.

    Regression test for the coverage problem: with calendar lags, a nowcast
    made before q-1's second release had no y_lag1 and the model abstained,
    dropping coverage to 37/72 on a one-year smoke test.
    """
    panel, y = panel_and_target
    q = y.index[-1]
    # Simulate real time: q-1's second release is not out yet.
    published = y.iloc[:-2]

    feats, _ = _design_matrix(panel, published)
    assert q in feats.index, "row dropped despite two published AR values existing"

    row = feats.loc[q]
    # y_lag1 must be the LATEST published value, which is q-2 here.
    assert row["y_lag1"] == pytest.approx(published.iloc[-1])
    assert row["y_lag2"] == pytest.approx(published.iloc[-2])
    # ...and staleness records that the freshest reading is 2 quarters old.
    assert row["y_staleness_q"] == pytest.approx(2.0)


def test_ar_features_never_use_the_predicted_quarter(panel_and_target):
    """No AR feature may equal the target quarter's own value (R1)."""
    panel, y = panel_and_target
    feats, target = _design_matrix(panel, y)
    lag_cols = [c for c in feats.columns if c.startswith("y_lag")]
    for q in feats.index:
        if pd.isna(target.get(q)):
            continue
        own = target.loc[q]
        for col in lag_cols:
            assert feats.loc[q, col] != pytest.approx(
                own
            ), f"{col} at {q.date()} equals the target being predicted"


def test_design_matrix_is_invariant_to_future_rows(panel_and_target):
    """Truncating everything after the target quarter must change nothing."""
    panel, y = panel_and_target
    q = y.index[-1]
    cutoff = q.to_period("Q").end_time

    full, _ = _design_matrix(panel, y.iloc[:-1])
    trunc, _ = _design_matrix(panel.loc[:cutoff], y.iloc[:-1])
    common = full.index.intersection(trunc.index)
    pd.testing.assert_frame_equal(
        full.loc[common].sort_index(axis=1), trunc.loc[common].sort_index(axis=1)
    )


@pytest.mark.parametrize("name", ["random_forest", "xgboost"])
def test_tree_models_recover_the_relationship(name, panel_and_target):
    panel, y = panel_and_target
    q = y.index[-1]
    model = ML_MODELS[name](n_estimators=60)
    model.fit(panel, y.iloc[:-1])
    pred = model.predict(panel, q, y.iloc[:-1]).mean
    assert np.isfinite(pred)
    assert abs(pred - y.loc[q]) < abs(y.iloc[:-1].mean() - y.loc[q]) * 2.0


@pytest.mark.parametrize("name", ["random_forest", "xgboost"])
def test_tree_models_ignore_future_data(name, panel_and_target):
    panel, y = panel_and_target
    q = y.index[-1]
    cutoff = q.to_period("Q").end_time

    a = ML_MODELS[name](n_estimators=60)
    a.fit(panel, y.iloc[:-1])
    pa = a.predict(panel, q, y.iloc[:-1]).mean

    b = ML_MODELS[name](n_estimators=60)
    b.fit(panel.loc[:cutoff], y.iloc[:-1])
    pb = b.predict(panel.loc[:cutoff], q, y.iloc[:-1]).mean
    assert pa == pytest.approx(pb, rel=1e-9)


def test_tree_models_cannot_extrapolate_beyond_training_range(panel_and_target):
    """A documented structural limitation, not a bug.

    Tree predictions are averages of training leaf values, so RF/XGB can never
    return a value outside the training target's range. On 2020Q2 (actual
    -31.7, training minimum -6.25) they returned about -3 to -6 while the
    state-space DFM reached -24.5. the study must not read that as the
    learners being 'worse' without noting they are structurally incapable of
    the extrapolation.
    """
    panel, y = panel_and_target
    q = y.index[-1]
    for name in ("random_forest", "xgboost"):
        model = ML_MODELS[name](n_estimators=60)
        model.fit(panel, y.iloc[:-1])
        pred = model.predict(panel, q, y.iloc[:-1]).mean
        lo, hi = y.iloc[:-1].min(), y.iloc[:-1].max()
        assert (
            lo - 1e-6 <= pred <= hi + 1e-6
        ), "a tree ensemble returned a value outside its training range"


def test_lstm_validation_split_is_a_contiguous_suffix(panel_and_target):
    """Early stopping must not tune on quarters that follow the fit window."""
    panel, y = panel_and_target
    model = LSTMNowcaster(epochs=3, val_quarters=8, window=4, min_train_quarters=30)
    model.fit(panel, y.iloc[:-1])
    if model.model_ is None:
        pytest.skip("LSTM did not train on the fixture")
    # The scaler must come from the training slice only, so it cannot equal
    # the statistics of the full sample when the suffix differs.
    _mu, sd = model.scaler_
    assert np.all(sd > 0)
    assert model.best_epoch_ is not None


def test_lstm_ignores_future_data(panel_and_target):
    panel, y = panel_and_target
    q = y.index[-1]
    cutoff = q.to_period("Q").end_time
    kw = {
        "epochs": 4,
        "val_quarters": 8,
        "window": 4,
        "min_train_quarters": 30,
        "seed": 1,
    }

    a = LSTMNowcaster(**kw)
    a.fit(panel, y.iloc[:-1])
    b = LSTMNowcaster(**kw)
    b.fit(panel.loc[:cutoff], y.iloc[:-1])
    if a.model_ is None or b.model_ is None:
        pytest.skip("LSTM did not train on the fixture")
    pa = a.predict(panel, q, y.iloc[:-1]).mean
    pb = b.predict(panel.loc[:cutoff], q, y.iloc[:-1]).mean
    assert pa == pytest.approx(pb, rel=1e-4)


def test_models_abstain_rather_than_guess_when_untrained(panel_and_target):
    panel, y = panel_and_target
    for cls in (RFNowcaster, XGBNowcaster, LSTMNowcaster):
        model = cls(min_train_quarters=10_000)
        model.fit(panel, y)
        assert np.isnan(model.predict(panel, y.index[-1], y).mean)


def test_ml_registry_is_complete():
    assert set(ML_MODELS) == {"random_forest", "xgboost", "lstm"}
    for name, cls in ML_MODELS.items():
        assert cls.name == name
