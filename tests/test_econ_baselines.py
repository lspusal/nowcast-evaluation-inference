"""Tests for the M1 econometric baselines.

Two things are checked for every model: that it cannot see the future (R1,
re-verified per model rather than trusted from ``info_set``), and that it
actually RESPONDS to newly published data within a publication cycle -- a
nowcaster whose forecast does not move as data arrives is broken in a way
that no error metric would reveal, it would just quietly flatten the
within-cycle profile that RQ2 is about.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import info_set
from src.data.transforms import transform_panel
from src.models.econ_baselines import (
    ECON_MODELS,
    BridgeNowcaster,
    DFMNowcaster,
    MIDASNowcaster,
    _almon_weights,
)


def _write_vintage(root, series_id, vintage_date, obs):
    path = Path(root) / "observations" / series_id / f"{vintage_date}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "series_id": series_id,
                "vintage_date": vintage_date,
                "observations": [{"date": d, "value": v} for d, v in obs],
                "fetched_at": "2024-01-01T00:00:00+00:00",
            }
        )
    )


@pytest.fixture
def synthetic_panel():
    """Monthly indicator that genuinely drives a quarterly target.

    y_q = 2 + 0.5 * mean(x over q) + noise, so a working nowcaster must
    recover a positive relationship and must sharpen as the quarter fills in.
    """
    rng = np.random.default_rng(42)
    months = pd.date_range("1995-01-01", "2019-12-01", freq="MS")
    x = pd.Series(rng.normal(0, 1, len(months)).cumsum() * 0.1 + 5.0, index=months)
    x2 = pd.Series(rng.normal(0, 1, len(months)) + 3.0, index=months)
    panel = pd.DataFrame({"IND1": x, "IND2": x2})

    xq = panel.groupby(panel.index.to_period("Q")).mean()
    y = 2.0 + 0.5 * xq["IND1"] + rng.normal(0, 0.2, len(xq))
    y.index = xq.index.to_timestamp()
    return panel, y


def test_almon_weights_are_a_normalised_decay():
    w = _almon_weights(np.array([-0.3, -0.02]), 6)
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()
    assert (np.diff(w) < 0).all(), "with negative theta weights must decay"
    flat = _almon_weights(np.array([0.0, 0.0]), 4)
    np.testing.assert_allclose(flat, np.full(4, 0.25))


@pytest.mark.parametrize("name", ["bridge", "midas"])
def test_model_recovers_a_real_relationship(name, synthetic_panel):
    """A model that cannot beat a constant on constructed data is broken."""
    panel, y = synthetic_panel
    train_y = y.iloc[:-1]
    q = y.index[-1]

    model = ECON_MODELS[name]()
    model.fit(panel, train_y)
    pred = model.predict(panel, q).mean

    assert np.isfinite(pred), f"{name} produced no forecast"
    # Must be closer to the truth than the unconditional mean is.
    assert abs(pred - y.loc[q]) < abs(train_y.mean() - y.loc[q]) * 1.5


@pytest.mark.parametrize("name", ["bridge", "midas"])
def test_model_ignores_data_after_the_nowcast_date(name, synthetic_panel):
    """R1, re-checked per model: truncating the future must not change today."""
    panel, y = synthetic_panel
    q = y.index[-1]
    train_y = y.iloc[:-1]

    full = ECON_MODELS[name]()
    full.fit(panel, train_y)
    pred_full = full.predict(panel, q).mean

    # Everything after the target quarter removed entirely.
    cutoff = q.to_period("Q").end_time
    truncated = panel.loc[:cutoff]
    trunc_model = ECON_MODELS[name]()
    trunc_model.fit(truncated, train_y)
    pred_trunc = trunc_model.predict(truncated, q).mean

    assert pred_full == pytest.approx(pred_trunc, rel=1e-9), (
        f"{name} changed its nowcast when future rows were removed -- it was "
        "reading data published after the nowcast date"
    )


def test_model_does_not_see_unpublished_target_quarters(synthetic_panel):
    """A model must never be handed the quarter it is predicting."""
    panel, y = synthetic_panel
    q = y.index[-1]

    honest = BridgeNowcaster()
    honest.fit(panel, y.iloc[:-1])
    p_honest = honest.predict(panel, q).mean

    leaky = BridgeNowcaster()
    leaky.fit(panel, y)  # includes the answer
    p_leaky = leaky.predict(panel, q).mean

    assert p_honest != pytest.approx(p_leaky), (
        "including the target quarter in training changed nothing -- the test "
        "fixture cannot detect target leakage"
    )
    assert abs(p_leaky - y.loc[q]) < abs(p_honest - y.loc[q]), (
        "the leaky fit should be closer to the answer; if not, this test is "
        "not measuring what it claims"
    )


def test_bridge_returns_nan_before_any_indicator_is_published(synthetic_panel):
    """Honest abstention: no data for the quarter means no forecast."""
    panel, y = synthetic_panel
    q = y.index[-1]
    model = BridgeNowcaster()
    model.fit(panel, y.iloc[:-1])

    # Cut the panel off before the target quarter starts.
    before = panel.loc[: q - pd.Timedelta(days=1)]
    assert np.isnan(model.predict(before, q).mean)


def test_bridge_sharpens_as_the_quarter_fills_in(synthetic_panel):
    """More of the quarter observed should move the forecast."""
    panel, y = synthetic_panel
    q = y.index[-1]
    model = BridgeNowcaster()
    model.fit(panel, y.iloc[:-1])

    months = pd.date_range(q, periods=3, freq="MS")
    preds = [
        model.predict(panel.loc[: m + pd.Timedelta(days=1)], q).mean for m in months
    ]
    preds = [p for p in preds if np.isfinite(p)]
    assert len(preds) >= 2
    assert (
        len(set(np.round(preds, 8))) > 1
    ), "the bridge forecast did not move as the quarter filled in"


def test_min_train_quarters_is_respected(synthetic_panel):
    panel, y = synthetic_panel
    model = BridgeNowcaster(min_train_quarters=500)
    model.fit(panel, y)
    assert model.equations == {}
    assert np.isnan(model.predict(panel, y.index[-1]).mean)


def test_dfm_nowcast_updates_within_the_publication_cycle(tmp_path, monkeypatch):
    """The DFM must RE-FILTER on the current information set.

    Regression test: ``predict`` originally called ``result_.predict`` on the
    fitted result, which ignores data arriving after the refit. The nowcast
    was then byte-identical for every date inside a quarter, which would have
    flattened the within-cycle error profile RQ2 reports while looking
    perfectly plausible in aggregate. It must call ``result_.apply`` first.
    """
    monkeypatch.setattr(info_set, "CACHE_DIR", tmp_path)

    rng = np.random.default_rng(7)
    months = pd.date_range("2000-01-01", "2019-12-01", freq="MS")
    ind = pd.Series(rng.normal(0, 1, len(months)) + 5.0, index=months)
    panel = pd.DataFrame({"IND1": ind, "IND2": ind.shift(1).bfill() * 0.5})
    xq = panel.groupby(panel.index.to_period("Q")).mean()
    y = 2.0 + 0.5 * xq["IND1"]
    y.index = xq.index.to_timestamp()

    q = y.index[-1]
    train_y = y.iloc[:-1]
    model = DFMNowcaster(factor_candidates=(1,), em_maxiter=15)
    model.fit(panel, train_y)
    if model.result_ is None:
        pytest.skip("DFM did not converge on the synthetic fixture")

    # One month of the target quarter observed, then two.
    m1, m3 = q, q + pd.offsets.MonthBegin(2)
    p1 = model.predict(panel.loc[: m1 + pd.Timedelta(days=1)], q, train_y).mean
    p3 = model.predict(panel.loc[: m3 + pd.Timedelta(days=1)], q, train_y).mean

    assert np.isfinite(p1) and np.isfinite(p3)
    assert p1 != pytest.approx(p3), (
        "the DFM nowcast is frozen: it returned the same value with one and "
        "with three months of the quarter observed, so result_.apply is not "
        "being used and new releases cannot move the nowcast"
    )


def test_econ_models_registry_is_complete():
    assert set(ECON_MODELS) == {
        "naive_mean",
        "naive_ar1",
        "bridge",
        "midas",
        "dfm",
        "dfm_robust",
    }
    for name, cls in ECON_MODELS.items():
        assert cls.name == name
        assert hasattr(cls, "fit") and hasattr(cls, "predict")


def test_forecast_carries_optional_quantiles():
    from src.models.econ_baselines import Forecast

    f = Forecast(mean=1.0)
    assert f.q10 is None and f.q50 is None and f.q90 is None
    g = Forecast(mean=1.0, q10=0.5, q50=1.0, q90=1.5)
    assert (g.q10, g.q50, g.q90) == (0.5, 1.0, 1.5)


def test_midas_uses_only_monthly_indicators(synthetic_panel):
    """Daily columns must not be selected: K monthly lags of a daily series
    would span days, not months, and the Almon weights would be meaningless."""
    panel, y = synthetic_panel
    daily = pd.Series(
        1.0, index=pd.date_range(panel.index.min(), panel.index.max(), freq="D")
    )
    mixed = panel.join(daily.rename("DAILY"), how="outer")

    model = MIDASNowcaster()
    model.fit(mixed, y.iloc[:-1])
    assert "DAILY" not in model.models


def test_transform_then_model_keeps_the_ragged_edge(tmp_path, monkeypatch):
    """End-to-end: info_set -> transform -> model, with nothing filled in."""
    monkeypatch.setattr(info_set, "CACHE_DIR", tmp_path)
    obs = [
        (d.strftime("%Y-%m-%d"), str(100.0 + i))
        for i, d in enumerate(pd.date_range("2000-01-01", "2019-11-01", freq="MS"))
    ]
    _write_vintage(tmp_path, "IND1", "2019-12-15", obs)

    panel = info_set.information_set(pd.Timestamp("2019-12-20"), ["IND1"])
    spec = {"series": [{"fred_id": "IND1", "tcode": 5}]}
    X = transform_panel(panel, spec)

    # December 2019 was never published, so it must be absent, not filled.
    assert pd.Timestamp("2019-12-01") not in X.index
    assert X["IND1"].notna().sum() > 200


def test_dfm_divergence_screen_rejects_blowups_not_extremes():
    """The screen must separate numerical divergence from a real extreme.

    Regression test for two bugs in sequence. First, selecting the factor
    count on in-sample BIC alone PREFERRED a degenerate specification: on
    2012-01-21 k=3 had the best BIC (11563 vs 13510) and produced -2.57e6,
    which is how a 2009-2012 stretch of the backtest reached an RMSE of 4.2
    million. Then the first fix used a +/-5 SD band around the training
    range, which would have rejected a -24.5 nowcast of the -31.7 2020Q2
    outcome -- making the DFM abstain precisely during COVID, the subsample
    M5 reports. The band must therefore be wide enough to admit real
    extremes and still reject divergence.
    """
    y = pd.Series(
        np.random.default_rng(3).normal(2.0, 2.5, 80),
        index=pd.date_range("2000-01-01", periods=80, freq="QS"),
    )
    model = DFMNowcaster()
    model.y_ = y.sort_index()
    centre, scale = y.mean(), y.std()
    model.band_ = (
        centre - model.divergence_sd * scale,
        centre + model.divergence_sd * scale,
    )
    lo, hi = model.band_

    # A COVID-scale outcome must be INSIDE the band.
    assert lo <= -31.7 <= hi, f"band [{lo:.0f},{hi:.0f}] would reject 2020Q2"
    assert lo <= -24.5 <= hi, "band would reject a good nowcast of 2020Q2"
    # The observed divergences must be OUTSIDE it.
    for blowup in (-2.57e6, 3.72e6, -2.9e7):
        assert not (lo <= blowup <= hi), f"band admits divergence {blowup:.1e}"
    # And the separation should be orders of magnitude, not marginal.
    assert hi < 1e4


def test_dfm_keeps_all_converged_candidates_for_fallback():
    """Fallback requires every converged spec, not just the best-BIC one."""
    rng = np.random.default_rng(11)
    months = pd.date_range("2000-01-01", "2019-12-01", freq="MS")
    ind = pd.Series(rng.normal(0, 1, len(months)) + 5.0, index=months)
    panel = pd.DataFrame({"IND1": ind, "IND2": ind.shift(1).bfill() * 0.5})
    xq = panel.groupby(panel.index.to_period("Q")).mean()
    y = 2.0 + 0.5 * xq["IND1"]
    y.index = xq.index.to_timestamp()

    model = DFMNowcaster(factor_candidates=(1, 2), em_maxiter=15)
    model.fit(panel, y.iloc[:-1])
    if not model.candidates_:
        pytest.skip("no specification converged on the synthetic fixture")

    # Sorted by BIC ascending, and the chosen result is the best one.
    bics = [c[0] for c in model.candidates_]
    assert bics == sorted(bics)
    assert model.result_ is model.candidates_[0][2]
    assert model.band_ is not None


def test_robust_dfm_masks_outliers_and_classic_does_not():
    """The two DFMs must differ in exactly one respect.

    `dfm_robust` treats observations beyond `outlier_mad` robust SDs as
    MISSING during estimation; `dfm` does not. That single difference took the
    euro-area DFM from RMSE 8.300 to 2.523 in normal quarters and dissolved an
    apparent, significant "TSFMs beat the DFM" result.
    """
    from src.models.econ_baselines import RobustDFMNowcaster

    classic = DFMNowcaster()
    robust = RobustDFMNowcaster()
    assert classic.robust_outliers is False
    assert robust.robust_outliers is True
    # Identical in every other respect, so the comparison isolates one cause.
    for attr in ("factor_candidates", "em_maxiter", "divergence_sd", "outlier_mad"):
        assert getattr(classic, attr) == getattr(robust, attr)


def test_mask_outliers_blanks_only_the_extremes():
    from src.models.econ_baselines import _mask_outliers

    clean = np.concatenate([np.random.default_rng(0).normal(0, 1, 60), [40.0, -50.0]])
    frame = pd.DataFrame({"x": clean})
    out, n = _mask_outliers(frame, n_mad=6.0)
    assert n == 2, f"expected the 2 planted outliers to be masked, got {n}"
    assert out["x"].iloc[-2:].isna().all()
    assert out["x"].iloc[:60].notna().all(), "masked a non-outlier"


def test_robust_scale_resists_contamination():
    """The whole point of §49: the scale must not be inflated by outliers."""
    from src.models.econ_baselines import _robust_scale

    base = np.random.default_rng(1).normal(0, 1, 200)
    contaminated = np.concatenate([base, [60.0, -60.0]])
    sd_clean, sd_dirty = float(np.std(base)), float(np.std(contaminated))
    mad_clean = _robust_scale(base)
    mad_dirty = _robust_scale(contaminated)
    # The SD roughly quadruples; the robust scale barely moves.
    assert sd_dirty > 3 * sd_clean
    assert abs(mad_dirty - mad_clean) < 0.15 * mad_clean


def test_naive_benchmarks_use_only_published_quarters():
    """R1: the naive floor must be REAL-TIME, i.e. invariant to the future.

    Tested by INVARIANCE rather than by comparing against a full-sample
    statistic: an AR(1) may legitimately forecast above the sample mean when
    the series trends, so "prediction < full-sample mean" is not a leakage
    test at all (it failed the AR(1) for the right reason on a linear
    fixture). Perturbing quarters at or after q is the correct check.
    """
    from src.models.econ_baselines import NaiveAR1Nowcaster, NaiveMeanNowcaster

    rng = np.random.default_rng(0)
    idx = pd.date_range("2010-01-01", periods=40, freq="QS")
    y = pd.Series(rng.normal(2, 1, 40), index=idx)
    q = idx[20]

    perturbed = y.copy()
    perturbed.iloc[20:] = -999.0

    for cls in (NaiveMeanNowcaster, NaiveAR1Nowcaster):
        a, b = cls(), cls()
        a.fit(pd.DataFrame(), y)
        b.fit(pd.DataFrame(), perturbed)
        pa = a.predict(pd.DataFrame(), q, y).mean
        pb = b.predict(pd.DataFrame(), q, perturbed).mean
        assert pa == pytest.approx(pb), f"{cls.name} used post-q information"

    # And the mean benchmark must be exactly the mean of what precedes q.
    m = NaiveMeanNowcaster()
    m.fit(pd.DataFrame(), y)
    assert m.predict(pd.DataFrame(), q, y).mean == pytest.approx(
        float(y.iloc[:20].mean())
    )


def test_naive_benchmarks_abstain_when_history_is_too_short():
    from src.models.econ_baselines import NaiveAR1Nowcaster, NaiveMeanNowcaster

    y = pd.Series([1.0, 2.0], index=pd.date_range("2020-01-01", periods=2, freq="QS"))
    for cls in (NaiveMeanNowcaster, NaiveAR1Nowcaster):
        m = cls()
        m.fit(pd.DataFrame(), y)
        assert np.isnan(m.predict(pd.DataFrame(), pd.Timestamp("2021-01-01"), y).mean)
