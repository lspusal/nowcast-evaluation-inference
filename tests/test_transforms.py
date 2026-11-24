"""Tests for FRED-MD transformation codes (M0 acceptance)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import info_set
from src.data.transforms import apply_tcode, transform_panel


def test_tcode_identities():
    """Verify each tcode on a known series (hand-computed expected values)."""
    x = pd.Series([1.0, 2.0, 4.0, 8.0])  # doubles every period
    ln2 = np.log(2.0)

    expected = {
        1: [1.0, 2.0, 4.0, 8.0],
        2: [np.nan, 1.0, 2.0, 4.0],
        3: [np.nan, np.nan, 1.0, 2.0],
        4: [0.0, ln2, 2 * ln2, 3 * ln2],
        5: [np.nan, ln2, ln2, ln2],
        6: [np.nan, np.nan, 0.0, 0.0],
        7: [np.nan, np.nan, 0.0, 0.0],
    }
    for tcode, exp in expected.items():
        got = apply_tcode(x, tcode).to_numpy()
        np.testing.assert_allclose(got, exp, atol=1e-12, equal_nan=True)

    with pytest.raises(ValueError):
        apply_tcode(x, 99)


def test_log_guard_nonpositive(caplog):
    """Non-positive values under log tcodes become NaN with a warning."""
    x = pd.Series([2.0, 0.0, -1.0, 4.0])
    with caplog.at_level("WARNING"):
        out = apply_tcode(x, 4)
    assert np.isnan(out.iloc[1])
    assert np.isnan(out.iloc[2])
    assert out.iloc[0] == pytest.approx(np.log(2.0))
    assert out.iloc[3] == pytest.approx(np.log(4.0))
    assert any("non-positive" in r.message for r in caplog.records)


def test_transform_panel_uses_spec_tcodes():
    df = pd.DataFrame({"A": [1.0, 2.0, 4.0], "B": [10.0, 20.0, 30.0]})
    spec = {"series": [{"fred_id": "A", "tcode": 5}, {"fred_id": "B", "tcode": 2}]}
    out = transform_panel(df, spec)
    np.testing.assert_allclose(
        out["A"].to_numpy(), [np.nan, np.log(2.0), np.log(2.0)], equal_nan=True
    )
    np.testing.assert_allclose(
        out["B"].to_numpy(), [np.nan, 10.0, 10.0], equal_nan=True
    )


def test_transform_panel_survives_mixed_frequency_index():
    """A monthly series on a daily-dominated index must NOT be annihilated.

    Regression test for the mixed-frequency bug: positional .diff() on the
    union index differenced each monthly observation against a NaN daily
    neighbour, silently destroying 66-94% of every non-daily series.
    """
    daily = pd.date_range("2020-01-01", "2020-03-31", freq="D")
    monthly = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
    idx = daily.union(monthly)

    panel = pd.DataFrame(index=idx, columns=["DLY", "MON"], dtype=float)
    panel.loc[daily, "DLY"] = np.arange(1.0, len(daily) + 1.0)
    panel.loc[monthly, "MON"] = [100.0, 200.0, 400.0]  # doubles each month

    spec = {"series": [{"fred_id": "DLY", "tcode": 2}, {"fred_id": "MON", "tcode": 5}]}
    out = transform_panel(panel, spec)

    # The monthly series keeps one transformed value per observation after
    # the first (its own diff), not zero.
    mon = out["MON"].dropna()
    assert len(mon) == 2, f"monthly series annihilated: only {len(mon)} values left"
    np.testing.assert_allclose(mon.to_numpy(), [np.log(2.0), np.log(2.0)])

    # Differences are between consecutive MONTHLY observations, not rows.
    assert out.loc[pd.Timestamp("2020-02-01"), "MON"] == pytest.approx(np.log(2.0))

    # The ragged edge is preserved: no value appears where none was observed.
    assert (
        out.loc[pd.Timestamp("2020-02-15"), "MON"]
        != out.loc[pd.Timestamp("2020-02-15"), "MON"]
    )  # NaN

    # And the daily series is unharmed.
    assert out["DLY"].notna().sum() == len(daily) - 1


def test_transform_panel_refuses_silent_passthrough():
    """A column with no tcode would leak raw levels into a transformed panel."""
    df = pd.DataFrame({"A": [1.0, 2.0], "GDPC1": [100.0, 101.0]})
    spec = {"series": [{"fred_id": "A", "tcode": 5}]}
    with pytest.raises(ValueError, match="no tcode in spec"):
        transform_panel(df, spec)

    out = transform_panel(df, spec, allow_untransformed=True)
    np.testing.assert_allclose(out["GDPC1"].to_numpy(), [100.0, 101.0])


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


def test_target_is_second_release(tmp_path, monkeypatch):
    """Target series uses the 2nd vintage of each quarter (R2).

    Synthetic ALFRED-style vintage cache (network-free, deterministic):
    quarter final levels are 100 (Q3'19), 101 (Q4'19), 102 (Q1'20), 103
    (Q2'20); each quarter is released as advance (final - 0.5), second
    (final - 0.2), then third/final, one month apart.
    """
    monkeypatch.setattr(info_set, "CACHE_DIR", tmp_path)

    quarters = {
        "2019-07-01": (100.0, "2019-10-30", "2019-11-27", "2019-12-27"),
        "2019-10-01": (101.0, "2020-01-30", "2020-02-27", "2020-03-27"),
        "2020-01-01": (102.0, "2020-04-29", "2020-05-28", "2020-06-25"),
        "2020-04-01": (103.0, "2020-07-30", "2020-08-27", "2020-09-24"),
    }
    all_vintage_dates = sorted({d for v in quarters.values() for d in v[1:]})

    for vd in all_vintage_dates:
        obs = []
        for q, (final, advance_d, second_d, third_d) in quarters.items():
            if vd >= third_d:
                val = final
            elif vd >= second_d:
                val = final - 0.2
            elif vd >= advance_d:
                val = final - 0.5
            else:
                continue
            obs.append((q, str(val)))
        _write_vintage(tmp_path, "TESTGDP", vd, obs)

    target = info_set.second_release_target("TESTGDP")
    q4 = pd.Timestamp("2019-10-01")
    ratio = (101.0 - 0.2) / 100.0

    # DEFAULT is BEA's compounded annualization -- the number BEA actually
    # publishes, which is what R2 means by "segunda estimación oficial"
    #.
    assert info_set.DEFAULT_GROWTH_FORMULA == "compounded"
    assert target.loc[q4] == pytest.approx(100.0 * (ratio**4 - 1.0))

    # The log approximation remains available but is NOT the default.
    log_form = info_set.second_release_target("TESTGDP", growth_formula="log")
    assert log_form.loc[q4] == pytest.approx(
        400.0 * (np.log(101.0 - 0.2) - np.log(100.0))
    )
    assert log_form.loc[q4] != pytest.approx(target.loc[q4])

    with pytest.raises(ValueError):
        info_set.second_release_target("TESTGDP", growth_formula="pct_change")

    records = info_set._second_release_records("TESTGDP")
    rec_q4 = next(r for r in records if r["quarter"] == q4)
    assert rec_q4["second_release_vintage"] == pd.Timestamp("2020-02-27")

    saved = info_set.save_second_release_target(
        "TESTGDP", out_path=str(tmp_path / "target_second_release.csv")
    )
    assert (tmp_path / "target_second_release.csv").exists()
    assert len(saved) == len(target)


def test_target_excludes_quarters_truncated_by_vintage_history(tmp_path, monkeypatch):
    """R2: a quarter already published before our vintage history begins has
    no observable advance release, so its "second release" would actually be a
    later annual/benchmark revision - exactly what docs/01_data_spec.md §3
    says to avoid. Such quarters must be dropped, not fabricated.

    On the real GDPC1 cache this guard removed 178 of 316 rows, e.g. 1947Q2,
    which had been assigned a "second release" vintage of 1997-05-07.
    """
    monkeypatch.setattr(info_set, "CACHE_DIR", tmp_path)

    # OLD was already published before the cache starts; its value simply sits
    # there unchanged, then moves at a benchmark revision years later.
    # NEW is genuinely released inside the covered window.
    _write_vintage(
        tmp_path,
        "TRUNC",
        "2010-01-29",
        [
            ("2009-04-01", "100.0"),
            ("2009-07-01", "101.0"),
        ],
    )
    _write_vintage(
        tmp_path,
        "TRUNC",
        "2010-02-26",
        [
            ("2009-04-01", "100.0"),
            ("2009-07-01", "101.0"),
            ("2009-10-01", "102.0"),
        ],
    )
    _write_vintage(
        tmp_path,
        "TRUNC",
        "2010-03-26",
        [
            ("2009-04-01", "100.0"),
            ("2009-07-01", "101.0"),
            ("2009-10-01", "102.3"),
        ],
    )
    # Benchmark revision years later touches the old quarters.
    _write_vintage(
        tmp_path,
        "TRUNC",
        "2013-07-31",
        [
            ("2009-04-01", "110.0"),
            ("2009-07-01", "111.0"),
            ("2009-10-01", "112.0"),
        ],
    )

    quarters = {r["quarter"] for r in info_set._second_release_records("TRUNC")}

    # 2009Q2 was already published at the truncation boundary: its only later
    # "change" is the 2013 benchmark revision -> must be excluded.
    assert pd.Timestamp("2009-04-01") not in quarters
    # 2009Q3 likewise sits at the boundary.
    assert pd.Timestamp("2009-07-01") not in quarters
    # 2009Q4 is genuinely observed: advance 2010-02-26, second 2010-03-26.
    assert pd.Timestamp("2009-10-01") in quarters
    rec = next(
        r
        for r in info_set._second_release_records("TRUNC")
        if r["quarter"] == pd.Timestamp("2009-10-01")
    )
    assert rec["second_release_vintage"] == pd.Timestamp("2010-03-26")
    assert rec["release_gap_days"] == 28


def test_target_rejects_implausible_release_gap(tmp_path, monkeypatch):
    """An advance->second gap far from ~1 month means the ordinal is wrong.

    Real instance: 1995Q3's advance was delayed by the federal shutdown, so
    the next change point was a rebasing 84 days later (5544.6 -> 6763.2).
    """
    monkeypatch.setattr(info_set, "CACHE_DIR", tmp_path)
    _write_vintage(tmp_path, "GAP", "2010-01-01", [("2009-07-01", "50.0")])
    _write_vintage(
        tmp_path,
        "GAP",
        "2010-01-29",
        [
            ("2009-07-01", "50.0"),
            ("2009-10-01", "100.0"),
        ],
    )
    # Next change for 2009Q4 comes 84 days later -- a rebasing, not a release.
    _write_vintage(
        tmp_path,
        "GAP",
        "2010-04-23",
        [
            ("2009-07-01", "50.0"),
            ("2009-10-01", "126.0"),
        ],
    )

    quarters = {r["quarter"] for r in info_set._second_release_records("GAP")}
    assert pd.Timestamp("2009-10-01") not in quarters

    # Without the guard it would have been (wrongly) accepted.
    loose = {
        r["quarter"] for r in info_set._second_release_records("GAP", strict=False)
    }
    assert pd.Timestamp("2009-10-01") in loose
