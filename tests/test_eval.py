"""Tests for the evaluation pipeline (metrics, DM, MCS, NY Fed validation).

These guard the three rules that decide whether the study's numbers mean
anything: only genuine forecasts are scored, abstentions stay visible, and
models are compared on identical samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval import metrics as M
from src.eval.dm_test import _newey_west_variance, dm_table, dm_test
from src.eval.mcs import mcs
from src.eval.nyfed import nyfed_validation_table


@pytest.fixture
def preds_and_target():
    """Two models over 3 quarters x 4 dates; model 'good' beats 'bad'."""
    rng = np.random.default_rng(0)
    quarters = pd.to_datetime(["2015-01-01", "2015-04-01", "2015-07-01"])
    target = pd.Series([2.0, 3.0, 1.0], index=quarters, name="actual")

    rows = []
    for q in quarters:
        for k in range(4):
            date = q + pd.Timedelta(days=7 * k)
            horizon = (q.to_period("Q").end_time - date).days
            for model, noise in (("good", 0.2), ("bad", 2.0)):
                rows.append(
                    {
                        "nowcast_date": date,
                        "target_quarter": q,
                        "yhat": target[q] + rng.normal(0, noise),
                        "model": model,
                        "variant": "baseline",
                        "run_id": "test",
                        "horizon_days": horizon,
                        "target_already_published": False,
                    }
                )
    return pd.DataFrame(rows), target


def test_prepare_excludes_already_published_pairs(preds_and_target):
    """A pair whose answer was public is not a forecast and must be dropped."""
    preds, target = preds_and_target
    preds = preds.copy()
    preds.loc[preds.index[:6], "target_already_published"] = True

    prepared = M.prepare(preds, target)
    assert len(prepared) == len(preds) - 6
    assert not prepared["target_already_published"].any()

    # ...but the caller can opt out explicitly.
    kept = M.prepare(preds, target, genuine_only=False)
    assert len(kept) == len(preds)


def test_coverage_report_makes_abstentions_visible(preds_and_target):
    preds, target = preds_and_target
    preds = preds.copy()
    mask = (preds["model"] == "good") & (preds.index % 2 == 0)
    preds.loc[mask, "yhat"] = np.nan

    cov = M.coverage_report(M.prepare(preds, target))
    assert cov.loc["good", "coverage"] < 1.0
    assert cov.loc["bad", "coverage"] == 1.0
    assert cov.loc["good", "n_answered"] < cov.loc["good", "n_pairs"]


def test_common_sample_prevents_reward_for_abstaining(preds_and_target):
    """A model that skips hard dates must not be scored on an easier sample."""
    preds, target = preds_and_target
    preds = preds.copy()
    # 'good' abstains on the two worst dates for itself.
    prepared_all = M.prepare(preds, target)
    worst = (
        prepared_all[prepared_all["model"] == "good"]
        .assign(abserr=lambda d: d["error"].abs())
        .nlargest(2, "abserr")[["nowcast_date", "target_quarter"]]
    )
    for _, r in worst.iterrows():
        sel = (
            (preds["model"] == "good")
            & (preds["nowcast_date"] == r["nowcast_date"])
            & (preds["target_quarter"] == r["target_quarter"])
        )
        preds.loc[sel, "yhat"] = np.nan

    common = M.common_sample(M.prepare(preds, target))
    counts = common.groupby("model").size()
    assert counts.nunique() == 1, "models must be scored on identical pairs"
    # The dropped dates are gone for BOTH models, not just the abstainer.
    for _, r in worst.iterrows():
        sub = common[
            (common["nowcast_date"] == r["nowcast_date"])
            & (common["target_quarter"] == r["target_quarter"])
        ]
        assert sub.empty


def test_error_table_relative_to_benchmark(preds_and_target):
    preds, target = preds_and_target
    et = M.error_table(preds, target, benchmark="good")
    allrows = et[et["block"] == "all"].set_index("model")
    assert allrows.loc["good", "rmse"] < allrows.loc["bad", "rmse"]
    assert allrows.loc["good", "rmse_rel_benchmark"] == pytest.approx(1.0)
    assert allrows.loc["bad", "rmse_rel_benchmark"] > 1.0


def test_cycle_block_boundaries():
    assert M.cycle_block(90) == "early"
    assert M.cycle_block(61) == "early"
    assert M.cycle_block(60) == "mid"
    assert M.cycle_block(1) == "mid"
    assert M.cycle_block(0) == "late"
    assert M.cycle_block(-30) == "late"


def test_newey_west_variance_grows_with_positive_autocorrelation():
    """HAC must inflate the variance of an autocorrelated differential.

    This is the whole reason the DM test here needs HAC: ~17 weekly nowcasts
    share a target quarter, so the loss differential is serially dependent.
    """
    rng = np.random.default_rng(5)
    e = rng.normal(0, 1, 500)
    ar = np.zeros(500)
    for i in range(1, 500):
        ar[i] = 0.8 * ar[i - 1] + e[i]

    iid_var = _newey_west_variance(e, 0)
    hac_var = _newey_west_variance(ar, 10)
    naive_var = _newey_west_variance(ar, 0)
    assert hac_var > naive_var * 2, "HAC failed to account for autocorrelation"
    assert iid_var == pytest.approx(_newey_west_variance(e, 0))


def test_dm_test_sign_and_symmetry():
    rng = np.random.default_rng(1)
    good = pd.Series(rng.normal(0, 1, 300))
    bad = pd.Series(rng.normal(0, 2, 300))
    s1, p1 = dm_test(good, bad, hac_lags=4)
    s2, p2 = dm_test(bad, good, hac_lags=4)
    assert s1 < 0 < s2, "negative statistic must mean the first model is better"
    assert s1 == pytest.approx(-s2)
    assert p1 == pytest.approx(p2)


def test_dm_test_identical_errors_cannot_reject():
    e = pd.Series(np.random.default_rng(2).normal(0, 1, 100))
    stat, p = dm_test(e, e, hac_lags=4)
    assert stat == 0.0
    assert p == 1.0


def test_dm_hln_correction_is_conservative():
    """The small-sample correction must shrink the statistic, not inflate it."""
    rng = np.random.default_rng(4)
    a = pd.Series(rng.normal(0, 1, 40))
    b = pd.Series(rng.normal(0, 1.6, 40))
    s_corrected, p_corrected = dm_test(a, b, hac_lags=4, hln_correction=True)
    s_raw, p_raw = dm_test(a, b, hac_lags=4, hln_correction=False)
    assert abs(s_corrected) < abs(s_raw)
    assert p_corrected > p_raw, "correction must not make rejection easier"


def test_dm_table_requires_known_benchmark(preds_and_target):
    preds, target = preds_and_target
    common = M.common_sample(M.prepare(preds, target))
    with pytest.raises(ValueError, match="benchmark"):
        dm_table(common, benchmark="nonexistent")

    tab = dm_table(common, benchmark="good", hac_lags=2)
    assert set(tab["model"]) == {"bad"}
    assert (tab["benchmark"] == "good").all()


def test_mcs_retains_only_the_best_when_gaps_are_large():
    rng = np.random.default_rng(7)
    n = 400
    losses = pd.DataFrame(
        {
            "best": rng.normal(0, 1, n) ** 2,
            "mid": rng.normal(0, 1.5, n) ** 2,
            "worst": rng.normal(0, 4, n) ** 2,
        }
    )
    out = mcs(losses, alpha=0.10, n_boot=800).set_index("model")
    assert out.loc["best", "in_mcs"]
    assert not out.loc["worst", "in_mcs"]
    # p-values are bounded and monotone in mean loss ordering.
    assert 0.0 <= out["mcs_pvalue"].min() <= out["mcs_pvalue"].max() <= 1.0


def test_mcs_keeps_everything_when_models_are_indistinguishable():
    """Equally good models must all survive -- the M1 outcome."""
    rng = np.random.default_rng(8)
    n = 200
    losses = pd.DataFrame({k: rng.normal(0, 1, n) ** 2 for k in ("a", "b", "c")})
    out = mcs(losses, alpha=0.10, n_boot=800)
    assert out["in_mcs"].all()


def test_mcs_rejects_degenerate_input():
    with pytest.raises(ValueError):
        mcs(pd.DataFrame({"only": [1.0] * 50}))
    with pytest.raises(ValueError):
        mcs(pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]}))


def test_nyfed_validation_table_flags_the_rho_threshold():
    """The acceptance flag must track the documented 0.8 target."""
    dates = pd.date_range("2016-01-01", periods=60, freq="W")
    base = np.linspace(-2, 6, 60)
    matched = pd.DataFrame(
        {
            "nowcast_date": dates,
            "target_quarter": dates.to_period("Q").to_timestamp(),
            "yhat": base + np.random.default_rng(0).normal(0, 0.2, 60),
            "nyfed": base,
            "regime": "realtime_v1",
            "forecast_date": dates,
            "gap_days": 0,
            "horizon_days": 30,
        }
    )
    tab = nyfed_validation_table(matched).set_index("regime")
    assert tab.loc["realtime_v1", "rho_pearson"] > 0.9
    assert bool(tab.loc["realtime_v1", "meets_target_rho_0.8"])

    # A regime with no relationship must fail the flag.
    noise = matched.copy()
    noise["yhat"] = np.random.default_rng(1).normal(0, 3, 60)
    noise["regime"] = "realtime_v2"
    tab2 = nyfed_validation_table(noise).set_index("regime")
    assert not bool(tab2.loc["realtime_v2", "meets_target_rho_0.8"])


def test_load_target_rejects_unknown_growth_formula(tmp_path):
    path = tmp_path / "target.csv"
    pd.DataFrame(
        {"quarter": ["2020-01-01"], "growth_compounded": [1.0], "growth_log": [1.0]}
    ).to_csv(path, index=False)
    assert M.load_target(path, "compounded").iloc[0] == 1.0
    with pytest.raises(ValueError, match="no column"):
        M.load_target(path, "nonsense")


def test_dm_test_refuses_an_implicit_bandwidth():
    """The h-1 convention gives a bandwidth of ZERO for a nowcast, i.e. no
    autocorrelation correction at all on an overlapping sample. Defaulting to
    it silently was a landmine, so it now raises."""
    import numpy as np
    import pytest

    from src.eval.dm_test import dm_test

    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=200))
    b = pd.Series(rng.normal(size=200))
    with pytest.raises(ValueError, match="hac_lags must be given explicitly"):
        dm_test(a, b)


def test_dm_table_derives_its_bandwidth_from_the_grid():
    """A hard-coded bandwidth is the error the accompanying paper is about,
    so dm_table computes one from its own sample's overlap when not told."""
    import numpy as np

    from src.eval.dm_test import dm_table, hac_lags_for_overlap

    rng = np.random.default_rng(1)
    quarters = pd.date_range("2010-01-01", periods=20, freq="QS")
    rows = []
    for q in quarters:
        for k in range(10):  # ten nowcast dates per target quarter
            date = q + pd.Timedelta(days=7 * k)
            for model, scale in (("good", 1.0), ("bad", 2.0)):
                rows.append(
                    {
                        "nowcast_date": date,
                        "target_quarter": q,
                        "model": model,
                        "error": scale * rng.normal(),
                        "block": "all",
                    }
                )
    prepared = pd.DataFrame(rows)
    table = dm_table(prepared, benchmark="good", by_block=False)
    assert not table.empty
    # The derived bandwidth must reflect the ten-per-quarter overlap, not 4.
    assert hac_lags_for_overlap(10) == 10
    explicit = dm_table(prepared, benchmark="good", hac_lags=10, by_block=False)
    assert np.allclose(
        table["dm_stat_weekly_hac"].to_numpy(float),
        explicit["dm_stat_weekly_hac"].to_numpy(float),
    ), "the derived bandwidth does not match the overlap implied by the grid"


# --------------------------------------------------------------------------
# Validation of the estimator the study's central claim rests on
# --------------------------------------------------------------------------
# The study's thesis is a statement about the size of a test. If our own
# implementation of that test were wrong, the whole size study would be an
# artifact of our code rather than of the design, so the estimator is
# validated against an external reference and against its own nominal size
#.


def test_newey_west_matches_statsmodels_hac():
    """The Bartlett-weighted long-run variance must equal the HAC variance of
    the mean that statsmodels computes for an intercept-only regression."""
    import numpy as np

    sm = pytest.importorskip("statsmodels.api")
    from src.eval.dm_test import _newey_west_variance

    rng = np.random.default_rng(11)
    d = rng.normal(size=400)
    d[1:] += 0.5 * d[:-1]  # give it autocorrelation to correct
    design = np.ones((len(d), 1))
    for lag in (1, 4, 17):
        fitted = sm.OLS(d, design).fit(
            cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False}
        )
        reference = float(fitted.cov_params()[0, 0])
        ours = _newey_west_variance(d, lag) / len(d)
        assert ours == pytest.approx(
            reference, rel=1e-10
        ), f"HAC({lag}): ours {ours:.10f} vs statsmodels {reference:.10f}"


def test_newey_west_reduces_to_the_sample_variance_at_zero_lags():
    import numpy as np

    from src.eval.dm_test import _newey_west_variance

    rng = np.random.default_rng(12)
    for n in (50, 200, 1000):
        d = rng.normal(size=n)
        assert _newey_west_variance(d, 0) == pytest.approx(
            float(((d - d.mean()) ** 2).mean()), rel=1e-12
        )


def test_hln_correction_is_the_documented_factor_at_h_one():
    import numpy as np

    from src.eval.dm_test import dm_test

    rng = np.random.default_rng(13)
    n = 100
    a = pd.Series(rng.normal(size=n))
    b = pd.Series(a + 0.1 * rng.normal(size=n))
    corrected, _ = dm_test(a, b, hac_lags=0, hln_correction=True)
    raw, _ = dm_test(a, b, hac_lags=0, hln_correction=False)
    assert corrected / raw == pytest.approx(np.sqrt((n - 1) / n), rel=1e-12)


def test_dm_test_has_nominal_size_without_overlap():
    """The control that licenses the size study: with one observation per
    target and no overlap, the test must be correctly sized. Part A of the
    study measures 6.5% at m=1 independently; this reaches the same place
    from a different construction."""
    import numpy as np

    from src.eval.dm_test import dm_test

    rng = np.random.default_rng(14)
    for lag, ceiling in ((0, 0.09), (4, 0.10)):
        rejections = 0
        reps = 1500
        for _ in range(reps):
            a = pd.Series(rng.normal(size=82))
            b = pd.Series(rng.normal(size=82))  # same law: the null holds
            _, p = dm_test(a, b, hac_lags=lag)
            rejections += p < 0.05
        size = rejections / reps
        assert 0.02 < size < ceiling, f"HAC({lag}) size {size:.3f} at nominal 0.05"


def test_dm_test_detects_a_real_difference():
    """Size without power would be satisfied by a test that never rejects."""
    import numpy as np

    from src.eval.dm_test import dm_test

    rng = np.random.default_rng(15)
    rejections = 0
    reps = 300
    for _ in range(reps):
        good = pd.Series(rng.normal(size=82))
        bad = pd.Series(rng.normal(scale=2.0, size=82))
        _, p = dm_test(good, bad, hac_lags=0)
        rejections += p < 0.05
    assert rejections / reps > 0.9


def test_quarterly_tests_have_nominal_size():
    """Both quarterly tests, under a null with one date per target quarter."""
    import numpy as np

    from src.eval.dm_test import dm_test_by_quarter

    rng = np.random.default_rng(16)
    quarters = pd.date_range("2005-01-01", periods=82, freq="QS")
    index = pd.MultiIndex.from_arrays(
        [quarters, quarters], names=["nowcast_date", "target_quarter"]
    )
    rej_t = rej_sign = 0
    reps = 1200
    for _ in range(reps):
        a = pd.Series(rng.normal(size=82), index=index)
        b = pd.Series(rng.normal(size=82), index=index)
        out = dm_test_by_quarter(a, b)
        rej_t += out["p_t"] < 0.05
        rej_sign += out["p_sign"] < 0.05
    assert 0.02 < rej_t / reps < 0.09, f"quarterly t size {rej_t / reps:.3f}"
    assert 0.01 < rej_sign / reps < 0.07, f"sign size {rej_sign / reps:.3f}"


def test_loading_predictions_excludes_the_ablation_runs(tmp_path):
    """An ablation must not be pooled with the run it is an ablation of.

    The ablation writes ``{model}__{ablation}.parquet`` carrying the SAME
    ``model`` value, distinguished only by ``variant``. Anything selecting an
    arm by ``model`` pooled the real run with the deliberately leaky ex-post
    panel: the factor model's relative RMSE fell from 0.705 to 0.631 and its
    quarter count rose from 78 to 81, because the ablation answers quarters
    the real run abstains on. Both numbers looked unremarkable.
    """
    import pandas as pd

    from src.eval.metrics import load_predictions

    preds = tmp_path / "preds"
    preds.mkdir()
    base = pd.DataFrame(
        {
            "nowcast_date": pd.to_datetime(["2020-01-01", "2020-01-08"]),
            "target_quarter": pd.to_datetime(["2020-01-01"] * 2),
            "yhat": [1.0, 2.0],
            "model": ["dfm", "dfm"],
            "variant": ["baseline", "baseline"],
        }
    )
    base.to_parquet(preds / "dfm.parquet", index=False)
    leaky = base.assign(variant="baseline+balanced_panel", yhat=[9.0, 9.0])
    leaky.to_parquet(preds / "dfm__balanced_panel.parquet", index=False)

    out = load_predictions(tmp_path)
    assert len(out) == 2, "the ablation rows were pooled with the real run"
    assert set(out["variant"]) == {"baseline"}
    assert out["yhat"].tolist() == [1.0, 2.0]

    both = load_predictions(tmp_path, include_ablations=True)
    assert len(both) == 4
