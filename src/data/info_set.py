"""Real-time information set construction (ragged edge).

Implements docs/01_data_spec.md §4 and docs/02_methodology.md §2.
This module is the ONLY place where the information set is built (R3).

Core contract
-------------
``information_set(t, series_ids) -> pd.DataFrame`` returns, for nowcast date
``t``, a (obs_date x series) matrix built by taking, for each series, the
ALFRED snapshot at the latest cached vintage_date <= t. Because an ALFRED
snapshot already only contains what was published as of that vintage, this
directly reproduces the ragged edge: an obs_date not yet published at ``t``
is simply absent (NaN) after joining.

Target construction (R2)
------------------------
``second_release_target(...)`` builds the QoQ SAAR growth of GDPC1 using the
SECOND official vintage of each reference quarter. "Second official vintage
of quarter Q" is defined as the vintage_date at which Q's value changes for
the SECOND time (the first change is the advance/first release). Growth is
then computed from the levels of Q and Q-1 as they both read in THAT SAME
vintage snapshot (not from Q-1's own second release, which may already be
stale by the time Q's second release is out) — this is the standard
real-time convention (see docs/01_data_spec.md §3) and is documented here
because the spec text alone is ambiguous about which snapshot to use.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.alfred_client import CACHE_DIR

logger = logging.getLogger(__name__)

# Plausible advance->second release gap, in days. Anything far outside the
# band means the "second release" was not identified correctly (a delayed
# release, an identical republish, or a benchmark/comprehensive revision).
#
# REGION-SPECIFIC. The default is calibrated on
# BEA: advance -> second is ~28 days (measured min 21 / median 28 / max 39
# over 138 US quarters). EUROSTAT publishes on a different cadence --
# preliminary flash (~t+30d), flash (~t+45d), then the regular estimate --
# so the euro-area advance->second gap clusters at 13-15 days and the BEA
# band silently rejected 20 of 41 valid quarters. The band is therefore a
# parameter, not a constant.
RELEASE_GAP_MIN_DAYS = 21
RELEASE_GAP_MAX_DAYS = 45


def _series_from_payload(payload: dict) -> pd.Series:
    obs = payload["observations"]
    if not obs:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([o["date"] for o in obs])
    vals = [
        float(o["value"]) if o["value"] not in (".", "", None) else float("nan")
        for o in obs
    ]
    return pd.Series(vals, index=idx).sort_index()


_VINTAGE_CACHE: dict[tuple, pd.DataFrame] = {}


def _cache_signature(obs_dir: Path) -> tuple:
    """Identity of a series' cache directory: file count + newest mtime.

    Any download that adds or rewrites a vintage changes this, so the memo
    below cannot serve stale data even while a download is running in
    parallel or a test monkeypatches CACHE_DIR to a temp directory.
    """
    paths = list(obs_dir.glob("*.json"))
    if not paths:
        return (0, 0.0)
    return (len(paths), max(p.stat().st_mtime for p in paths))


def clear_vintage_cache() -> None:
    """Drop the in-process vintage-matrix and release-history memos."""
    _VINTAGE_CACHE.clear()
    _RELEASE_CACHE.clear()


def vintage_matrix(series_id: str) -> pd.DataFrame:
    """Full (obs_date x vintage_date) matrix for a series from cache.

    Memoised per (directory, series, cache signature). A backtest calls this
    once per nowcast date via ``available_target``, and rebuilding GDPC1's
    matrix means re-reading 418 JSON files (~150 ms), i.e. minutes of pure
    I/O over a 1,400-date run -- and every later milestone multiplies that.
    The signature makes the memo self-invalidating rather than merely fast.
    """
    obs_dir = CACHE_DIR / "observations" / series_id
    if not obs_dir.exists():
        return pd.DataFrame()

    key = (str(CACHE_DIR), series_id, _cache_signature(obs_dir))
    hit = _VINTAGE_CACHE.get(key)
    if hit is not None:
        return hit

    columns = {}
    for path in sorted(obs_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        s = _series_from_payload(payload)
        if s.empty:
            continue
        columns[pd.Timestamp(payload["vintage_date"])] = s

    df = pd.DataFrame(columns)
    if not df.empty:
        df = df.sort_index(axis=0).sort_index(axis=1)
    # Bound the memo: a handful of series matrices is all a run needs, and
    # each can be tens of MB.
    if len(_VINTAGE_CACHE) > 64:
        _VINTAGE_CACHE.clear()
    _VINTAGE_CACHE[key] = df
    return df


def information_set(
    t: date,
    series_ids: list[str],
    target_id: str | None = None,
    target_quarter: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """(obs_date x series) panel as known at date ``t`` — ragged edge kept.

    R3: no forward-filling and no substitution with data whose vintage is
    after ``t``; a series/period not yet published at ``t`` stays NaN. Only
    cached vintage files dated <= t are ever read for a given series, so
    neither the values NOR THE SHAPE of the result can depend on what is
    cached for later vintages (e.g. a newly-published obs_date that only
    shows up in a future vintage must not appear as an extra NaN row here).

    SELF-LEAK OF THE TARGET (R1). The target series is part of the downloaded
    panel, and lagged GDP is a legitimate predictor -- but the quarter being
    predicted must not be readable. The second release of quarter Q-1 lands
    before many of Q's nowcast dates, so on ~38% of the 2005-2024 calendar the
    BACKCAST answer sits in the panel and a model that simply reads that
    column scores RMSE ~ 0. Pass ``target_id`` and ``target_quarter`` and every
    observation of the target dated at or after ``target_quarter`` is masked,
    leaving earlier quarters (the honest AR information) intact.
    """
    t = pd.Timestamp(t)
    columns = {}
    for series_id in series_ids:
        obs_dir = CACHE_DIR / "observations" / series_id
        eligible = (
            [p for p in obs_dir.glob("*.json") if pd.Timestamp(p.stem) <= t]
            if obs_dir.exists()
            else []
        )
        if not eligible:
            columns[series_id] = pd.Series(dtype=float)
            continue
        latest = max(eligible, key=lambda p: pd.Timestamp(p.stem))
        payload = json.loads(latest.read_text())
        series = _series_from_payload(payload)

        if target_id is not None and series_id == target_id:
            if target_quarter is None:
                raise ValueError(
                    "target_id given without target_quarter: the quarter being "
                    "predicted must be specified so it can be masked (R1)"
                )
            cutoff = pd.Timestamp(target_quarter)
            series = series.where(series.index < cutoff)

        columns[series_id] = series

    return pd.DataFrame(columns).sort_index()


def balanced_panel(
    series_ids: list[str],
    target_quarter: pd.Timestamp | str,
    target_id: str | None = None,
) -> pd.DataFrame:
    """EX-POST balanced panel for the H3 ablation (M5). DELIBERATELY LEAKY.

    Returns each series as it reads in its LATEST vintage, truncated at the
    end of ``target_quarter``: a complete, fully-revised quarter with no
    ragged edge. This is NOT a real-time information set and must never be
    used for a headline result -- it is the counterfactual upper bound that
    quantifies what the ragged edge costs (docs/00_master_plan.md M5 task 3).

    Two effects are removed at once and CANNOT be separated with this data:
    the ragged edge (months not yet published) and revisions (real-time vs
    final values). A "real-time values, complete quarter" panel would isolate
    the first, but that panel never existed -- those values were not published
    at the time. The confound is therefore inherent to the ablation, not a
    shortcut, and says so.
    """
    q_end = pd.Timestamp(target_quarter).to_period("Q").end_time
    columns = {}
    for series_id in series_ids:
        obs_dir = CACHE_DIR / "observations" / series_id
        files = sorted(obs_dir.glob("*.json")) if obs_dir.exists() else []
        if not files:
            columns[series_id] = pd.Series(dtype=float)
            continue
        latest = max(files, key=lambda p: pd.Timestamp(p.stem))
        series = _series_from_payload(json.loads(latest.read_text()))
        series = series.loc[series.index <= q_end]
        if target_id is not None and series_id == target_id:
            # Even the leaky panel must not hand over the answer itself.
            series = series.loc[series.index < pd.Timestamp(target_quarter)]
        columns[series_id] = series
    return pd.DataFrame(columns).sort_index()


def target_already_published(
    t: date, target_quarter: pd.Timestamp | str, series_id: str = "GDPC1"
) -> bool:
    """Was ``target_quarter``'s value already public at ``t``?

    An evaluation pair whose answer was already published is not a forecast.
    Use this to drop such pairs from the backcast horizon rather than
    reporting a spuriously low error for them.
    """
    t = pd.Timestamp(t)
    cutoff = pd.Timestamp(target_quarter)
    v = vintage_matrix(series_id)
    if v.empty or cutoff not in v.index:
        return False
    eligible = [c for c in v.columns if c <= t]
    if not eligible:
        return False
    return bool(pd.notna(v.loc[cutoff, max(eligible)]))


def available_target(
    t: date,
    series_id: str = "GDPC1",
    growth_formula: str | None = None,
    gap_min_days: int = RELEASE_GAP_MIN_DAYS,
    gap_max_days: int = RELEASE_GAP_MAX_DAYS,
) -> pd.Series:
    """The training target as it could be known at ``t`` (R1).

    R1 lists the *training target* explicitly: a model fitted at nowcast date
    ``t`` may only learn from target values already published at ``t``. The
    full second-release target series runs to the present, so handing it to
    ``fit()`` unfiltered trains every model on its own future -- a leak that
    no amount of care in the feature matrix would undo.

    Keeps quarters whose SECOND RELEASE vintage is <= ``t``, which is the same
    definition the evaluation target uses, so training and evaluation are
    consistent.
    """
    t = pd.Timestamp(t)
    formula = growth_formula or DEFAULT_GROWTH_FORMULA
    if formula not in GROWTH_FORMULAS:
        raise ValueError(f"growth_formula must be one of {GROWTH_FORMULAS}")

    records = _second_release_records(
        series_id, gap_min_days=gap_min_days, gap_max_days=gap_max_days
    )
    rows = {
        r["quarter"]: r[f"growth_{formula}"]
        for r in records
        if r["second_release_vintage"] <= t
    }
    if not rows:
        return pd.Series(dtype=float, name=series_id)
    return pd.Series(rows, name=series_id).sort_index()


def to_quarterly(
    panel: pd.DataFrame, how: str = "mean", min_obs: int = 1
) -> pd.DataFrame:
    """Aggregate a mixed-frequency ragged panel to quarterly averages.

    Each column is aggregated over the observations ACTUALLY PRESENT in the
    panel, so a partially-observed current quarter is summarised from just the
    months/days already published -- which is exactly the ragged edge, carried
    through to quarterly frequency rather than filled in. Quarters with fewer
    than ``min_obs`` observations for a series stay NaN for that series.

    R1/R3: this only ever reads what ``information_set`` returned, so it can
    introduce no information the caller did not already have.
    """
    if panel.empty:
        return pd.DataFrame()
    if how not in ("mean", "last", "sum"):
        raise ValueError(f"unsupported aggregation: {how!r}")

    grouper = panel.index.to_period("Q")
    counts = panel.groupby(grouper).count()
    agg = getattr(panel.groupby(grouper), how)()
    agg = agg.where(counts >= min_obs)
    agg.index = agg.index.to_timestamp()
    return agg.sort_index()


def _release_change_points(row: pd.Series) -> pd.Series:
    """Value at each vintage_date where ``row`` first takes a new value.

    ``row`` is indexed by vintage_date (ascending) for a single obs_date.
    """
    vals = row.dropna()
    if vals.empty:
        return vals
    return vals[vals.ne(vals.shift())]


def _previous_quarter(q: pd.Timestamp) -> pd.Timestamp:
    return (q.to_period("Q") - 1).start_time


GROWTH_FORMULAS = ("log", "compounded")
# R2 = the SECOND OFFICIAL ESTIMATE, i.e. the number BEA actually published,
# which uses compounded annualization. See.
DEFAULT_GROWTH_FORMULA = "compounded"

_RELEASE_CACHE: dict[tuple, list[dict]] = {}


def _second_release_records(
    series_id: str,
    strict: bool = True,
    gap_min_days: int = RELEASE_GAP_MIN_DAYS,
    gap_max_days: int = RELEASE_GAP_MAX_DAYS,
) -> list[dict]:
    """Records for each quarter's SECOND official release (R2).

    Memoised on the same cache signature as ``vintage_matrix``: the records
    do not depend on the nowcast date (``available_target`` only filters
    them), so a backtest would otherwise recompute the whole release history
    once per date -- and re-emit every skip warning each time.

    ``strict`` drops quarters whose release history is not trustworthy:

    * TRUNCATION. ALFRED's vintage history begins long after the series
      itself (GDPC1 levels start 1947 but its first vintage is 1991-12-04).
      For any quarter already published before that first vintage, the
      earliest "change point" is the truncation boundary, not the advance
      release -- so ``changes.index[1]`` is a later ANNUAL/BENCHMARK revision,
      exactly the vintage docs/01_data_spec.md §3 says to avoid. Without this
      guard 178 of 316 GDPC1 quarters were fabricated this way (e.g. 1947Q2
      assigned a "second release" vintage of 1997-05-07).
    * ANOMALOUS GAP. If the advance->second gap is outside
      [RELEASE_GAP_MIN_DAYS, RELEASE_GAP_MAX_DAYS] the ordinal is suspect
      (e.g. 1995Q3, where a shutdown-delayed advance made the next change
      point a rebasing 84 days later).

    Diagnostics (`n_changes`, `release_gap_days`, `first_release_vintage`)
    are kept on every record so a referee can audit the selection.
    """
    obs_dir = CACHE_DIR / "observations" / series_id
    key = (
        str(CACHE_DIR),
        series_id,
        strict,
        gap_min_days,
        gap_max_days,
        _cache_signature(obs_dir) if obs_dir.exists() else (0, 0.0),
    )
    cached = _RELEASE_CACHE.get(key)
    if cached is not None:
        return cached

    v = vintage_matrix(series_id)
    if v.empty:
        return []

    earliest_vintage = min(v.columns)
    records = []
    for q in v.index:
        changes = _release_change_points(v.loc[q])
        if len(changes) < 2:
            continue
        first_vintage = changes.index[0]
        second_vintage = changes.index[1]

        # Truncation guard: the quarter was already published before our
        # vintage history begins, so we never observed its real releases.
        if strict and first_vintage == earliest_vintage:
            logger.debug(
                "%s %s: skipped, first appearance is the cache truncation "
                "boundary (%s), not a real advance release",
                series_id,
                q.date(),
                earliest_vintage.date(),
            )
            continue

        gap_days = int((second_vintage - first_vintage).days)
        if strict and not (gap_min_days <= gap_days <= gap_max_days):
            # A gap of many months means the quarter's history was backfilled
            # into ALFRED long after the fact (structural, expected for
            # pre-1959 quarters) rather than mis-identified within a real
            # release cycle -- the latter is what deserves attention.
            structural = gap_days > 180
            logger.log(
                logging.DEBUG if structural else logging.WARNING,
                "%s %s: advance->second gap of %d days is outside [%d, %d]; "
                "the second release could not be identified reliably - skipped",
                series_id,
                q.date(),
                gap_days,
                gap_min_days,
                gap_max_days,
            )
            continue

        q_prev = _previous_quarter(q)
        if q_prev not in v.index:
            continue
        level_q = v.loc[q, second_vintage]
        level_prev = v.loc[q_prev, second_vintage]
        if pd.isna(level_q) or pd.isna(level_prev) or level_q <= 0 or level_prev <= 0:
            continue
        ratio = level_q / level_prev
        records.append(
            {
                "quarter": q,
                # As written in docs/01_data_spec.md §3: 400 * Δlog.
                "growth_log": 400.0 * np.log(ratio),
                # As actually published by BEA: ((x_t/x_{t-1})^4 - 1) * 100.
                # The two agree to ~0.05pp in normal quarters but diverge by
                # >6pp in 2008Q4/2020Q2 - see.
                "growth_compounded": 100.0 * (ratio**4 - 1.0),
                "first_release_vintage": first_vintage,
                "second_release_vintage": second_vintage,
                "release_gap_days": gap_days,
                "n_changes": len(changes),
                "level_q": level_q,
                "level_q_minus_1": level_prev,
            }
        )
    if len(_RELEASE_CACHE) > 32:
        _RELEASE_CACHE.clear()
    _RELEASE_CACHE[key] = records
    return records


def second_release_target(
    series_id: str = "GDPC1",
    growth_formula: str = DEFAULT_GROWTH_FORMULA,
    gap_min_days: int = RELEASE_GAP_MIN_DAYS,
    gap_max_days: int = RELEASE_GAP_MAX_DAYS,
) -> pd.Series:
    """QoQ SAAR growth evaluated at the second official release (R2).

    ``growth_formula`` selects the annualization convention: ``"log"`` is the
    400*Δlog form written in docs/01_data_spec.md §3; ``"compounded"`` is the
    ((x_t/x_{t-1})^4 - 1)*100 form BEA actually publishes. They differ
    materially in crisis quarters.
    """
    if growth_formula not in GROWTH_FORMULAS:
        raise ValueError(f"growth_formula must be one of {GROWTH_FORMULAS}")
    records = _second_release_records(
        series_id, gap_min_days=gap_min_days, gap_max_days=gap_max_days
    )
    if not records:
        return pd.Series(dtype=float)
    out = pd.DataFrame(records).set_index("quarter").sort_index()
    return out[f"growth_{growth_formula}"].rename(series_id)


def save_second_release_target(
    series_id: str = "GDPC1",
    out_path: str = "data/processed/target_second_release.csv",
    gap_min_days: int = RELEASE_GAP_MIN_DAYS,
    gap_max_days: int = RELEASE_GAP_MAX_DAYS,
) -> pd.DataFrame:
    """Build and persist the target with vintage metadata (docs/01_data_spec.md §3)."""
    from pathlib import Path

    records = _second_release_records(
        series_id, gap_min_days=gap_min_days, gap_max_days=gap_max_days
    )
    df = (
        pd.DataFrame(records).sort_values("quarter")
        if records
        else pd.DataFrame(
            columns=[
                "quarter",
                "growth_log",
                "growth_compounded",
                "second_release_vintage",
                "level_q",
                "level_q_minus_1",
            ]
        )
    )
    if not df.empty:
        df["quarter"] = df["quarter"].dt.date.astype(str)
        df["second_release_vintage"] = df["second_release_vintage"].dt.date.astype(str)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def nowcast_calendar(
    start: str,
    end: str,
    weeks_before_quarter_end: int = 13,
    series_id: str = "GDPC1",
) -> list[tuple[date, pd.Timestamp]]:
    """Weekly ``(nowcast_date, target_quarter)`` pairs (docs/01_data_spec.md §4).

    Per the spec, the cycle for target quarter ``Q`` runs weekly from
    ``weeks_before_quarter_end`` weeks BEFORE THE END OF Q -- which is what
    makes the window overlap the previous quarter's backcast -- through the
    eve of Q's advance release. An earlier implementation anchored on the
    release date instead, which silently deleted roughly the first four weeks
    of every cycle: precisely the "inicio del ciclo" bucket RQ2 is answered
    on. Q's own advance release date comes from ALFRED's vintage history
    (the release-calendar proxy).

    Returns PAIRS, not a flat list: the same calendar date belongs to
    different points of different quarters' cycles, and the prediction schema
    in docs/02_methodology.md §2.4 is keyed on
    ``(nowcast_date, target_quarter)``. Deduplicating to bare dates loses
    that mapping and forces every caller to reinvent it.

    ``start``/``end`` filter on the NOWCAST DATE, not on the target quarter.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    v = vintage_matrix(series_id)
    if v.empty:
        return []

    earliest_vintage = min(v.columns)
    pairs: list[tuple[date, pd.Timestamp]] = []
    for q in v.index:
        changes = _release_change_points(v.loc[q])
        if len(changes) < 1:
            continue
        release_date = changes.index[0]
        # A quarter already published before our vintage history begins has no
        # observable advance release; its "release date" is the truncation
        # boundary and would emit junk nowcast dates.
        if release_date == earliest_vintage:
            continue

        quarter_end = q.to_period("Q").end_time.normalize()
        cycle_start = quarter_end - timedelta(weeks=weeks_before_quarter_end)
        d = cycle_start
        while d < release_date:
            if start_ts <= d <= end_ts:
                pairs.append((d.date(), q))
            d += timedelta(weeks=1)

    return sorted(pairs)
