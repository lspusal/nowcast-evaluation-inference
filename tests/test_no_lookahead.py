"""CRITICAL anti-lookahead tests (R1/R3). Must pass before any model work (R6).

Implements the milestone M0 acceptance criteria.

Test strategy
-------------
These tests are the gate for all model work, so they are written to be able
to FAIL. Three kinds of assertion are used together, because an invariance
check alone is provably insufficient:

1. INVARIANCE — perturbing any vintage dated after ``t`` must not change the
   information set at ``t``. Catches "the future changed the answer".
2. ABSOLUTE (ground truth) — the information set at ``t`` must equal exactly
   the latest eligible vintage snapshot, cell for cell, INCLUDING its NaN
   mask. This is what catches a *uniform* leak such as ``.ffill()``, which
   invariance cannot see because both runs leak identically.
3. POSITIVE CONTROLS — deliberately leaky implementations are constructed and
   the assertions above must reject them. A gate that cannot fail is worse
   than no gate; these prove the gate bites.

The boundary convention (a vintage dated exactly ``t``) is pinned explicitly
rather than left implicit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import info_set
from src.data.impute import backward_impute


def _write_vintage(
    root: Path, series_id: str, vintage_date: str, obs: list[tuple[str, str]]
) -> None:
    path = root / "observations" / series_id / f"{vintage_date}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "series_id": series_id,
        "vintage_date": vintage_date,
        "observations": [{"date": d, "value": v} for d, v in obs],
        "fetched_at": "2024-01-01T00:00:00+00:00",
    }
    path.write_text(json.dumps(payload))


def _toy_vintages(root: Path) -> None:
    """Synthetic (obs_date x vintage_date) matrices with a ragged pattern.

    SERA gets revised at a later vintage; SERB lags SERA by a month and has an
    interior hole, so the panel at t has a genuine ragged edge whose NaN
    pattern a forward-fill would visibly destroy.
    """
    _write_vintage(
        root,
        "SERA",
        "2020-01-15",
        [("2019-11-01", "10.0"), ("2019-12-01", "11.0")],
    )
    _write_vintage(
        root,
        "SERA",
        "2020-02-15",
        [("2019-11-01", "10.5"), ("2019-12-01", "11.0"), ("2020-01-01", "12.0")],
    )
    # SERB has not published 2019-12-01 yet as of 2020-01-10: the ragged edge.
    _write_vintage(
        root,
        "SERB",
        "2020-01-10",
        [("2019-11-01", "1.0")],
    )
    _write_vintage(
        root,
        "SERB",
        "2020-02-20",
        [("2019-11-01", "1.0"), ("2019-12-01", "2.5"), ("2020-01-01", "3.0")],
    )


def _impute_fixture() -> pd.DataFrame:
    """Ragged frame including a column with NO history before row 4.

    Column C is the discriminating case: any fill of rows 0-3 can only have
    come from the future.
    """
    return pd.DataFrame(
        {
            "A": [1.0, None, 3.0, None, None, 6.0],
            "B": [None, 2.0, None, 4.0, 5.0, None],
            "C": [None, None, None, None, 7.0, 8.0],
        }
    )


# --------------------------------------------------------------------------
# 1. INVARIANCE: the future must not change the past
# --------------------------------------------------------------------------


def test_information_set_ignores_future_vintages(tmp_path, monkeypatch):
    """Perturbing post-t vintages must not change the information set at t."""
    monkeypatch.setattr(info_set, "CACHE_DIR", tmp_path)
    _toy_vintages(tmp_path)

    t = pd.Timestamp("2020-01-20")  # strictly between the two vintage dates
    before = info_set.information_set(t, ["SERA", "SERB"])

    _write_vintage(
        tmp_path,
        "SERA",
        "2020-02-15",
        [("2019-11-01", "-999.0"), ("2019-12-01", "-999.0"), ("2020-01-01", "-999.0")],
    )
    _write_vintage(
        tmp_path,
        "SERB",
        "2020-02-20",
        [("2019-11-01", "-999.0"), ("2019-12-01", "-999.0"), ("2020-01-01", "-999.0")],
    )
    # A brand-new, even-later vintage introducing a NEW obs_date must not leak
    # in, not even as an extra all-NaN row.
    _write_vintage(
        tmp_path,
        "SERA",
        "2020-03-01",
        [
            ("2019-11-01", "-1.0"),
            ("2019-12-01", "-1.0"),
            ("2020-01-01", "-1.0"),
            ("2020-02-01", "-1.0"),
        ],
    )

    after = info_set.information_set(t, ["SERA", "SERB"])
    pd.testing.assert_frame_equal(before, after)

    # Non-vacuity: perturbing a vintage AT OR BEFORE t *does* change it.
    _write_vintage(
        tmp_path,
        "SERA",
        "2020-01-15",
        [("2019-11-01", "-999.0"), ("2019-12-01", "-999.0")],
    )
    changed = info_set.information_set(t, ["SERA", "SERB"])
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(before, changed)


# --------------------------------------------------------------------------
# 2. ABSOLUTE: the panel must equal the eligible snapshot, NaN mask included
# --------------------------------------------------------------------------


def test_information_set_preserves_ragged_edge_exactly(tmp_path, monkeypatch):
    """R3: the panel at t must BE the latest eligible snapshot, holes and all.

    This is the assertion an invariance-only test cannot make. A uniform leak
    (ffill/bfill/interpolate/dropna) changes the NaN mask without breaking
    invariance, so the mask is checked against independently-stated truth.
    """
    monkeypatch.setattr(info_set, "CACHE_DIR", tmp_path)
    _toy_vintages(tmp_path)

    t = pd.Timestamp("2020-01-20")
    panel = info_set.information_set(t, ["SERA", "SERB"])

    # Ground truth for the vintages eligible at t (SERA@2020-01-15,
    # SERB@2020-01-10), written out by hand rather than derived from the code.
    truth = {
        "SERA": {"2019-11-01": 10.0, "2019-12-01": 11.0},
        "SERB": {"2019-11-01": 1.0},
    }
    for col, known in truth.items():
        for ts, value in panel[col].items():
            key = ts.strftime("%Y-%m-%d")
            if key in known:
                assert value == pytest.approx(known[key]), f"{col} @ {key}"
            else:
                assert np.isnan(
                    value
                ), f"{col} @ {key} must stay NaN (ragged edge), got {value!r}"

    # The specific hole a forward-fill would silently close.
    assert np.isnan(panel.loc[pd.Timestamp("2019-12-01"), "SERB"])


def test_information_set_never_invents_values(tmp_path, monkeypatch):
    """Every non-NaN cell must be traceable to the eligible cache file."""
    monkeypatch.setattr(info_set, "CACHE_DIR", tmp_path)
    _toy_vintages(tmp_path)

    t = pd.Timestamp("2020-01-20")
    panel = info_set.information_set(t, ["SERA", "SERB"])

    for col in panel.columns:
        eligible = [
            p
            for p in (tmp_path / "observations" / col).glob("*.json")
            if pd.Timestamp(p.stem) <= t
        ]
        latest = max(eligible, key=lambda p: pd.Timestamp(p.stem))
        payload = json.loads(latest.read_text())
        allowed = {o["date"]: float(o["value"]) for o in payload["observations"]}
        present = panel[col].dropna()
        assert len(present) == len(
            allowed
        ), f"{col}: {len(present)} values present, snapshot has {len(allowed)}"
        for ts, value in present.items():
            key = ts.strftime("%Y-%m-%d")
            assert key in allowed, f"{col} @ {key} is not in the eligible snapshot"
            assert value == pytest.approx(allowed[key])


# --------------------------------------------------------------------------
# 3. POSITIVE CONTROLS: prove the gate can actually fail
# --------------------------------------------------------------------------


def test_gate_rejects_forward_filled_ragged_edge(tmp_path, monkeypatch):
    """A leaky (ffill) information set MUST be caught by the ragged-edge check."""
    monkeypatch.setattr(info_set, "CACHE_DIR", tmp_path)
    _toy_vintages(tmp_path)
    t = pd.Timestamp("2020-01-20")

    honest = info_set.information_set(t, ["SERA", "SERB"])
    leaky = honest.ffill()  # the R3 violation

    assert np.isnan(honest.loc[pd.Timestamp("2019-12-01"), "SERB"])
    assert not np.isnan(
        leaky.loc[pd.Timestamp("2019-12-01"), "SERB"]
    ), "positive control broken: ffill did not change the NaN mask"
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(honest, leaky)


def test_gate_rejects_backfilled_imputation():
    """A leaky (bfill) imputation MUST be caught by the no-prior-history check."""
    df = _impute_fixture()
    honest = backward_impute(df)
    leaky = backward_impute(df).fillna(df.bfill())  # the R1 violation

    assert honest["C"].iloc[:4].isna().all()
    assert (
        leaky["C"].iloc[:4].notna().any()
    ), "positive control broken: bfill did not change the result"


# --------------------------------------------------------------------------
# Imputation (R1): backward-only
# --------------------------------------------------------------------------


def test_backward_impute_never_uses_future():
    """LOCF/expanding-mean imputation is invariant to future values."""
    df = _impute_fixture()
    k = 3  # perturb rows strictly after this index
    before = backward_impute(df)

    perturbed = df.copy()
    perturbed.iloc[k + 1 :] = -999.0
    after = backward_impute(perturbed)

    pd.testing.assert_frame_equal(before.iloc[: k + 1], after.iloc[: k + 1])


def test_backward_impute_leaves_pre_history_rows_missing():
    """ABSOLUTE check: a cell with no prior observation must stay NaN.

    This is what actually catches a backfill; the invariance test above does
    not, because a backfill leaks identically in both runs.
    """
    out = backward_impute(_impute_fixture())
    assert out["C"].iloc[:4].isna().all(), (
        "rows before a series' first observation were filled - only future "
        "data could have supplied them (R1 violation)"
    )
    assert out["C"].iloc[4] == pytest.approx(7.0)
    assert np.isnan(out["B"].iloc[0])


def test_backward_impute_truncation_invariance():
    """Imputing a prefix must equal imputing the whole frame and truncating."""
    df = _impute_fixture()
    full = backward_impute(df)
    for k in range(len(df)):
        prefix = backward_impute(df.iloc[: k + 1])
        pd.testing.assert_frame_equal(prefix, full.iloc[: k + 1])


# --------------------------------------------------------------------------
# Boundary convention
# --------------------------------------------------------------------------


def test_vintage_boundary_convention_is_pinned(tmp_path, monkeypatch):
    """A vintage dated EXACTLY t: pin the convention so it cannot drift.

    Convention: a nowcast dated ``t`` is made at
    the END of day ``t``, so a vintage published on ``t`` IS available. This
    test exists so that flipping the comparison is a visible, deliberate
    change rather than a silent one - it was previously unpinned, and
    `<=` -> `<` passed the whole suite unnoticed.
    """
    monkeypatch.setattr(info_set, "CACHE_DIR", tmp_path)
    _write_vintage(tmp_path, "SERA", "2020-01-15", [("2019-12-01", "1.0")])
    _write_vintage(tmp_path, "SERA", "2020-01-20", [("2019-12-01", "2.0")])

    exactly_on = info_set.information_set(pd.Timestamp("2020-01-20"), ["SERA"])
    day_before = info_set.information_set(pd.Timestamp("2020-01-19"), ["SERA"])

    assert exactly_on.loc[pd.Timestamp("2019-12-01"), "SERA"] == pytest.approx(2.0)
    assert day_before.loc[pd.Timestamp("2019-12-01"), "SERA"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------


def test_model_pipeline_no_leakage(tmp_path, monkeypatch):
    """End-to-end: stub model predictions at t unchanged by future data."""
    monkeypatch.setattr(info_set, "CACHE_DIR", tmp_path)
    _toy_vintages(tmp_path)

    def stub_predict(t: pd.Timestamp) -> float:
        """Trivial mean-predictor: mean of everything known at t."""
        panel = info_set.information_set(t, ["SERA", "SERB"])
        imputed = backward_impute(panel)
        return float(imputed.mean().mean())

    t1 = pd.Timestamp("2020-01-20")
    t2 = pd.Timestamp("2020-02-10")  # still before the second SERA/SERB vintage
    pred1_before = stub_predict(t1)
    pred2_before = stub_predict(t2)

    _write_vintage(
        tmp_path,
        "SERA",
        "2020-02-15",
        [("2019-11-01", "999.0"), ("2019-12-01", "999.0"), ("2020-01-01", "999.0")],
    )
    _write_vintage(
        tmp_path,
        "SERB",
        "2020-02-20",
        [("2019-11-01", "999.0"), ("2019-12-01", "999.0"), ("2020-01-01", "999.0")],
    )

    assert stub_predict(t1) == pytest.approx(pred1_before)
    assert stub_predict(t2) == pytest.approx(pred2_before)


# --------------------------------------------------------------------------
# Self-leak of the target (R1)
# --------------------------------------------------------------------------


def test_target_quarter_is_masked_from_its_own_information_set(tmp_path, monkeypatch):
    """The quarter being predicted must not be readable in the panel.

    Lagged target values are legitimate predictors, so only observations at or
    after ``target_quarter`` are masked - earlier quarters survive.
    """
    monkeypatch.setattr(info_set, "CACHE_DIR", tmp_path)
    # By 2020-08-27 the 2020Q2 value is published, so an unmasked panel hands
    # the model the very number it is meant to predict.
    _write_vintage(
        tmp_path,
        "GDPC1",
        "2020-08-27",
        [
            ("2019-10-01", "100.0"),
            ("2020-01-01", "98.0"),
            ("2020-04-01", "82.0"),
        ],
    )

    t = pd.Timestamp("2020-08-27")
    leaky = info_set.information_set(t, ["GDPC1"])
    assert leaky.loc[pd.Timestamp("2020-04-01"), "GDPC1"] == pytest.approx(82.0)

    masked = info_set.information_set(
        t, ["GDPC1"], target_id="GDPC1", target_quarter="2020-04-01"
    )
    assert np.isnan(
        masked.loc[pd.Timestamp("2020-04-01"), "GDPC1"]
    ), "the predicted quarter is still readable in its own information set"
    # Lagged quarters remain available as honest AR predictors.
    assert masked.loc[pd.Timestamp("2020-01-01"), "GDPC1"] == pytest.approx(98.0)
    assert masked.loc[pd.Timestamp("2019-10-01"), "GDPC1"] == pytest.approx(100.0)


def test_masking_requires_an_explicit_target_quarter(tmp_path, monkeypatch):
    monkeypatch.setattr(info_set, "CACHE_DIR", tmp_path)
    _write_vintage(tmp_path, "GDPC1", "2020-08-27", [("2020-04-01", "82.0")])
    with pytest.raises(ValueError, match="target_quarter"):
        info_set.information_set(
            pd.Timestamp("2020-08-27"), ["GDPC1"], target_id="GDPC1"
        )


def test_target_already_published_flags_non_forecasts(tmp_path, monkeypatch):
    """A pair whose answer was already public is not a forecast."""
    monkeypatch.setattr(info_set, "CACHE_DIR", tmp_path)
    _write_vintage(tmp_path, "GDPC1", "2020-07-30", [("2020-01-01", "98.0")])
    _write_vintage(
        tmp_path,
        "GDPC1",
        "2020-08-27",
        [("2020-01-01", "98.0"), ("2020-04-01", "82.0")],
    )

    # Before 2020Q2 was released: a genuine forecast.
    assert not info_set.target_already_published("2020-08-01", "2020-04-01")
    # After release: the "forecast" would just be reading the answer.
    assert info_set.target_already_published("2020-08-27", "2020-04-01")


# --------------------------------------------------------------------------
# Nowcast calendar (docs/01_data_spec.md §4)
# --------------------------------------------------------------------------


def test_nowcast_calendar_anchors_on_quarter_end_not_release(tmp_path, monkeypatch):
    """The cycle must start 13 weeks before the END OF Q, per the spec.

    Anchoring on the release date instead deletes the first ~4 weeks of every
    cycle - the "inicio del ciclo" bucket RQ2 is answered on.
    """
    monkeypatch.setattr(info_set, "CACHE_DIR", tmp_path)
    # 2019Q4 (ends 2019-12-31), advance release 2020-01-30.
    _write_vintage(tmp_path, "GDPC1", "2019-10-30", [("2019-07-01", "100.0")])
    _write_vintage(
        tmp_path,
        "GDPC1",
        "2020-01-30",
        [("2019-07-01", "100.0"), ("2019-10-01", "101.0")],
    )
    _write_vintage(
        tmp_path,
        "GDPC1",
        "2020-02-27",
        [("2019-07-01", "100.0"), ("2019-10-01", "101.3")],
    )

    pairs = info_set.nowcast_calendar("2019-01-01", "2020-12-31")
    q4 = pd.Timestamp("2019-10-01")
    q4_dates = sorted(d for d, q in pairs if q == q4)
    assert q4_dates, "no nowcast dates generated for 2019Q4"

    # 13 weeks (91 days) before quarter end 2019-12-31 == 2019-10-01, NOT 13
    # weeks before the 2020-01-30 release (which would be 2019-10-31 and would
    # drop the whole first month of the cycle).
    assert q4_dates[0] == pd.Timestamp("2019-10-01").date()
    assert q4_dates[0] < pd.Timestamp("2019-10-31").date()
    # ...and the cycle stops before the advance release.
    assert max(q4_dates) < pd.Timestamp("2020-01-30").date()


def test_nowcast_calendar_returns_target_quarter_pairs(tmp_path, monkeypatch):
    """The (nowcast_date, target_quarter) mapping must survive."""
    monkeypatch.setattr(info_set, "CACHE_DIR", tmp_path)
    _write_vintage(tmp_path, "GDPC1", "2019-10-30", [("2019-07-01", "100.0")])
    _write_vintage(
        tmp_path,
        "GDPC1",
        "2020-01-30",
        [("2019-07-01", "100.0"), ("2019-10-01", "101.0")],
    )
    _write_vintage(
        tmp_path,
        "GDPC1",
        "2020-02-27",
        [("2019-07-01", "100.0"), ("2019-10-01", "101.3")],
    )

    pairs = info_set.nowcast_calendar("2019-01-01", "2020-12-31")
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
    assert all(isinstance(q, pd.Timestamp) for _, q in pairs)
    # Every nowcast date precedes its own target quarter's release.
    assert len({q for _, q in pairs}) >= 1


def test_nowcast_calendar_filters_on_nowcast_date(tmp_path, monkeypatch):
    monkeypatch.setattr(info_set, "CACHE_DIR", tmp_path)
    _write_vintage(tmp_path, "GDPC1", "2019-10-30", [("2019-07-01", "100.0")])
    _write_vintage(
        tmp_path,
        "GDPC1",
        "2020-01-30",
        [("2019-07-01", "100.0"), ("2019-10-01", "101.0")],
    )
    _write_vintage(
        tmp_path,
        "GDPC1",
        "2020-02-27",
        [("2019-07-01", "100.0"), ("2019-10-01", "101.3")],
    )

    window = info_set.nowcast_calendar("2019-11-01", "2019-11-30")
    assert window, "expected some dates in the window"
    for d, _ in window:
        assert (
            pd.Timestamp("2019-11-01").date() <= d <= pd.Timestamp("2019-11-30").date()
        )


# --------------------------------------------------------------------------
# Training target must respect the vintage cutoff (R1)
# --------------------------------------------------------------------------


def _gdp_release_cache(root: Path) -> None:
    """Three quarters, each with advance/second/third releases a month apart."""
    quarters = {
        "2019-07-01": (100.0, "2019-10-30", "2019-11-27", "2019-12-27"),
        "2019-10-01": (101.0, "2020-01-30", "2020-02-27", "2020-03-27"),
        "2020-01-01": (102.0, "2020-04-29", "2020-05-28", "2020-06-25"),
    }
    all_vd = sorted({d for v in quarters.values() for d in v[1:]})
    for vd in all_vd:
        obs = []
        for q, (final, adv, sec, third) in quarters.items():
            if vd >= third:
                val = final
            elif vd >= sec:
                val = final - 0.2
            elif vd >= adv:
                val = final - 0.5
            else:
                continue
            obs.append((q, str(val)))
        _write_vintage(root, "GDPC1", vd, obs)


def test_available_target_excludes_unpublished_quarters(tmp_path, monkeypatch):
    """R1 names the TRAINING target: fit() must not see its own future.

    A model fitted at t may only learn from quarters whose second release was
    already out at t.
    """
    monkeypatch.setattr(info_set, "CACHE_DIR", tmp_path)
    _gdp_release_cache(tmp_path)

    # 2019Q4's second release is 2020-02-27; 2020Q1's is 2020-05-28.
    early = info_set.available_target("2020-03-01")
    assert pd.Timestamp("2019-10-01") in early.index
    assert pd.Timestamp("2020-01-01") not in early.index, (
        "a quarter whose second release had not happened yet leaked into the "
        "training target"
    )

    later = info_set.available_target("2020-06-01")
    assert pd.Timestamp("2020-01-01") in later.index

    # Monotone: the training target may only ever grow as t advances.
    assert set(early.index) <= set(later.index)


def test_available_target_is_invariant_to_future_vintages(tmp_path, monkeypatch):
    """Adding later vintages must not change what was trainable at t."""
    monkeypatch.setattr(info_set, "CACHE_DIR", tmp_path)
    _gdp_release_cache(tmp_path)

    before = info_set.available_target("2020-03-01")
    # A much later benchmark revision rewrites history.
    _write_vintage(
        tmp_path,
        "GDPC1",
        "2023-07-27",
        [("2019-07-01", "150.0"), ("2019-10-01", "151.0"), ("2020-01-01", "152.0")],
    )
    after = info_set.available_target("2020-03-01")
    pd.testing.assert_series_equal(before, after)


def test_to_quarterly_preserves_the_ragged_edge(tmp_path, monkeypatch):
    """Quarterly aggregation must summarise only what was published."""
    monkeypatch.setattr(info_set, "CACHE_DIR", tmp_path)
    # At this vintage only the first month of 2020Q1 is out.
    _write_vintage(
        tmp_path,
        "MON",
        "2020-02-15",
        [
            ("2019-10-01", "1.0"),
            ("2019-11-01", "2.0"),
            ("2019-12-01", "3.0"),
            ("2020-01-01", "10.0"),
        ],
    )
    panel = info_set.information_set(pd.Timestamp("2020-02-20"), ["MON"])
    q = info_set.to_quarterly(panel)

    # 2019Q4 is complete: mean(1,2,3) = 2.
    assert q.loc[pd.Timestamp("2019-10-01"), "MON"] == pytest.approx(2.0)
    # 2020Q1 is partial: the mean of the ONE published month, not an
    # interpolation and not NaN.
    assert q.loc[pd.Timestamp("2020-01-01"), "MON"] == pytest.approx(10.0)

    # min_obs lets a caller demand complete quarters instead.
    strict = info_set.to_quarterly(panel, min_obs=3)
    assert np.isnan(strict.loc[pd.Timestamp("2020-01-01"), "MON"])
    assert strict.loc[pd.Timestamp("2019-10-01"), "MON"] == pytest.approx(2.0)


def test_a_completion_arm_that_completed_nothing_abstains():
    """The single-factor arm passes a foundation-model-completed panel to a
    factor model. When the backend is unavailable it abstains inside the
    completer, which returns the panel uncompleted, and the factor model then
    produces a perfectly plausible nowcast from it.

    That is a comparison of the factor model with itself wearing the other
    arm's name, and it produced a full 1,403-row artifact at 100% coverage
    before anyone noticed. The arm must abstain instead.
    """
    import numpy as np
    import pandas as pd

    from src.models.econ_baselines import Forecast
    from src.models.tsfm_models import TSFMCompletedDFMNowcaster

    class _AlwaysAnswers:
        def predict(self, *a, **k):
            return Forecast(mean=1.234)

    arm = TSFMCompletedDFMNowcaster.__new__(TSFMCompletedDFMNowcaster)
    arm.name = "probe_completed_dfm"
    arm.dfm = _AlwaysAnswers()
    arm._complete_panel = lambda X, q: X
    panel = pd.DataFrame({"a": [1.0]})

    arm.n_completed_ = 0
    assert not np.isfinite(arm.predict(panel, "2020Q1").mean), (
        "a completion arm that filled no months returned a number; that is the "
        "factor model compared against itself"
    )

    arm.n_completed_ = 5
    assert np.isfinite(arm.predict(panel, "2020Q1").mean)


def test_a_resumed_run_discards_rows_written_by_other_model_code(tmp_path):
    """A checkpoint is only reusable if the same code wrote it.

    An arm whose panel completer was returning an empty context wrote 1,100
    abstentions to its checkpoint. The completer was fixed and the run
    resubmitted; it treated those 1,100 dates as already done and recomputed
    only the remaining 303. The job reported success and the output file was
    complete, correctly sized and 78% stale. Nothing in it said so.
    """
    import pandas as pd

    from src.experiments.run_backtest import _load_done_pairs

    path = tmp_path / "arm.parquet"
    rows = pd.DataFrame(
        {
            "nowcast_date": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]),
            "target_quarter": pd.to_datetime(["2020-01-01"] * 3),
            "yhat": [float("nan"), float("nan"), 1.5],
            "code_fingerprint": ["oldcode00000", "oldcode00000", "newcode11111"],
        }
    )
    rows.to_parquet(path, index=False)

    done = _load_done_pairs(path, "newcode11111")
    assert len(done) == 1, "rows from superseded code were treated as done"
    assert (pd.Timestamp("2020-01-15").date(), pd.Timestamp("2020-01-01")) in done

    # Without a fingerprint to check against, the old behaviour is preserved.
    assert len(_load_done_pairs(path)) == 3
