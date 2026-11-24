"""The declared config metadata must match the data actually on disk.

`freq` is not read by any model today (`transform_panel` works on each
series' own observed index), so a wrong label changes no result -- but it
WOULD the moment any code dispatches on frequency, which is exactly the trap
the design notes was written about. These tests make the config answerable to
the cache instead of to whoever edited it last.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

CACHE = Path("data/raw/fred/observations")
_EXPECTED_GAP = {"D": (1, 1), "W": (7, 7), "M": (28, 31), "Q": (89, 92)}


def _configs():
    return [p for p in Path("configs").glob("data*.yaml")]


def _measured_freq(series_id: str) -> str | None:
    d = CACHE / series_id
    files = sorted(d.glob("*.json")) if d.exists() else []
    if not files:
        return None
    obs = json.loads(files[-1].read_text())["observations"]
    if len(obs) < 8:
        return None
    idx = pd.to_datetime([o["date"] for o in obs])
    gaps = pd.Series(idx).diff().dt.days.dropna()
    if gaps.empty:
        return None
    modal = int(gaps.mode().iloc[0])
    for label, (lo, hi) in _EXPECTED_GAP.items():
        if lo <= modal <= hi:
            return label
    return f"{modal}d"


@pytest.mark.parametrize("config_path", _configs(), ids=lambda p: p.name)
def test_declared_freq_matches_the_cached_data(config_path):
    """Regression test for the design notes.

    `CPF3M` was labelled `freq: D` because four sibling spread/market series
    measured as daily and it was set by ANALOGY rather than measured. It is
    monthly (356 observations, 31-day modal gap). Anything inferred rather
    than measured belongs under a test.
    """
    cfg = yaml.safe_load(config_path.read_text())
    mismatches = []
    checked = 0
    for entry in cfg.get("series", []):
        sid, declared = entry["fred_id"], entry.get("freq")
        measured = _measured_freq(sid)
        if measured is None or declared is None:
            continue
        checked += 1
        if measured != declared:
            mismatches.append(f"{sid}: declared {declared}, measured {measured}")
    if checked == 0:
        pytest.skip(f"no cached series for {config_path.name}")
    assert not mismatches, "config freq disagrees with the data: " + "; ".join(
        mismatches
    )


@pytest.mark.parametrize("config_path", _configs(), ids=lambda p: p.name)
def test_declared_vintage_start_matches_the_cache(config_path):
    """`vintage_start` drives the baseline/extended tiering, so it must be real."""
    cfg = yaml.safe_load(config_path.read_text())
    wrong, checked = [], 0
    for entry in cfg.get("series", []):
        sid = entry["fred_id"]
        declared = entry.get("vintage_start")
        d = CACHE / sid
        files = sorted(d.glob("*.json")) if d.exists() else []
        if not declared or not files:
            continue
        checked += 1
        actual = pd.Timestamp(files[0].stem)
        if actual != pd.Timestamp(declared):
            wrong.append(f"{sid}: declared {declared}, cache starts {actual.date()}")
    if checked == 0:
        pytest.skip(f"no cached series for {config_path.name}")
    assert not wrong, "vintage_start disagrees with the cache: " + "; ".join(wrong)


def test_target_is_never_listed_as_a_regressor():
    """The target must not appear in `series` (self-leak, the design notes)."""
    for config_path in _configs():
        cfg = yaml.safe_load(config_path.read_text())
        target = cfg.get("target", {}).get("fred_id")
        ids = [e["fred_id"] for e in cfg.get("series", [])]
        assert target not in ids, f"{config_path.name}: target {target} is a regressor"


# --------------------------------------------------------------------------
# Vintage density: whether a region can support a second-release target AT ALL
# --------------------------------------------------------------------------
_MAX_MEDIAN_VINTAGE_GAP_DAYS = 60
_MIN_TARGET_QUARTERS = 20
_MIN_CONTIGUITY = 0.75

# Regions known NOT to support the R2 target, with the reason. Listing one
# here is a decision on the record, not a way to silence the test: the numbers
# are still asserted to be as bad as claimed.
_KNOWN_INSUFFICIENT = {"CLVMNACSCAB1GQDE": "Germany, the design notes"}


def _target_ids():
    for cfg_path in _configs():
        cfg = yaml.safe_load(cfg_path.read_text())
        target = (cfg or {}).get("target", {})
        if target.get("fred_id"):
            yield cfg_path, target["fred_id"], cfg.get("processed_dir")


@pytest.mark.parametrize("case", list(_target_ids()), ids=lambda c: c[1])
def test_target_vintage_density_supports_a_second_release(case):
    """A second-release target (R2) needs at least one ALFRED snapshot per
    release, so the first and second estimates are separable. Germany fails
    this at a median 69-day gap; the run had to be discarded AFTER it
    finished. This screen makes the failure visible before a run is launched.
    """
    cfg_path, series_id, _ = case
    d = CACHE / series_id
    if not d.exists() or not list(d.glob("*.json")):
        pytest.skip(f"{series_id} not cached")
    from src.data import info_set

    vm = info_set.vintage_matrix(series_id)
    if vm.shape[1] < 2:
        pytest.skip(f"{series_id}: only {vm.shape[1]} vintage(s) cached")
    gaps = pd.Series(vm.columns).diff().dropna().dt.days
    median_gap = float(gaps.median())
    if series_id in _KNOWN_INSUFFICIENT:
        # Assert the documented failure still holds; if the data improves,
        # this fires and the exemption should be removed, not widened.
        assert median_gap > _MAX_MEDIAN_VINTAGE_GAP_DAYS, (
            f"{series_id} ({_KNOWN_INSUFFICIENT[series_id]}) now has median gap "
            f"{median_gap:.0f}d -- coverage improved, revisit the exclusion"
        )
        return
    assert median_gap <= _MAX_MEDIAN_VINTAGE_GAP_DAYS, (
        f"{series_id} in {cfg_path.name}: median {median_gap:.0f}d between "
        f"vintages cannot separate a first from a second release"
    )


@pytest.mark.parametrize("case", list(_target_ids()), ids=lambda c: c[1])
def test_built_target_is_long_and_contiguous_enough_to_backtest(case):
    """A target with holes is worse than a short one: the backtest silently
    abstains and the abstentions look like model failures rather than missing
    data (which is exactly how §64 presented itself)."""
    cfg_path, series_id, processed_dir = case
    if not processed_dir:
        pytest.skip(f"{cfg_path.name} declares no processed_dir")
    path = Path(processed_dir) / "target_second_release.csv"
    if not path.exists():
        pytest.skip(f"{path} not built")
    q = pd.read_csv(path, parse_dates=["quarter"])["quarter"].sort_values()
    periods = q.dt.to_period("Q")
    span = int((periods.iloc[-1] - periods.iloc[0]).n) + 1
    contiguity = len(q) / span
    insufficient = series_id in _KNOWN_INSUFFICIENT
    if insufficient:
        assert len(q) < _MIN_TARGET_QUARTERS or contiguity < _MIN_CONTIGUITY, (
            f"{series_id} ({_KNOWN_INSUFFICIENT[series_id]}) now yields {len(q)} "
            f"quarters at {contiguity:.0%} contiguity -- revisit the exclusion"
        )
        return
    assert len(q) >= _MIN_TARGET_QUARTERS, f"{series_id}: only {len(q)} quarters"
    assert contiguity >= _MIN_CONTIGUITY, (
        f"{series_id}: {len(q)} quarters over a {span}-quarter span "
        f"({contiguity:.0%} contiguous) -- the backtest would mostly abstain"
    )


def test_cli_restriction_flags_reach_the_backtest(monkeypatch, tmp_path):
    """`--families` and `--ablation` must do what they say.

    Both were parsed and then dropped at the call site, so a run asking for
    one family silently refitted every model in the config. Nothing failed:
    the extra arms were simply computed, which cost time on every imputation
    draw and would have put an unrequested ablation in the output directory.
    """
    import sys

    import yaml

    from src.experiments import run_backtest

    cfg = {"data_config": "configs/data.yaml", "models": {"econ": ["dfm"]}}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))

    seen = {}

    def fake_run_stage_all(cfg, run_id, **kw):
        seen.update(kw)
        return {"ok": True}

    monkeypatch.setattr(run_backtest, "run_stage_all", fake_run_stage_all)
    monkeypatch.setattr(
        run_backtest, "write_manifest", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_backtest",
            "--config",
            str(path),
            "--stage",
            "all",
            "--families",
            "tsfm",
            "--ablation",
            "ragged_edge",
        ],
    )
    run_backtest.main()

    assert seen.get("families") == (
        "tsfm",
    ), f"--families did not reach the backtest; got {seen.get('families')!r}"
    assert (
        seen.get("ablation") == "ragged_edge"
    ), f"--ablation did not reach the backtest; got {seen.get('ablation')!r}"
