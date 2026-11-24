"""FRED-MD transformation codes (McCracken & Ng 2016).

Implements docs/01_data_spec.md §2 and milestone M0 task 2.
Transformations are applied PER VINTAGE on raw levels only.

tcode mapping (FRED-MD standard):
    1: x (no transform)
    2: Δx
    3: Δ²x
    4: log(x)
    5: Δlog(x)
    6: Δ²log(x)
    7: Δ(x_t / x_{t-1} - 1)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_log(s: pd.Series) -> pd.Series:
    """log(s), with non-positive values mapped to NaN (never silently coerced)."""
    bad = s <= 0
    n_bad = int(bad.sum())
    if n_bad:
        logger.warning(
            "%d non-positive value(s) encountered under a log tcode; set to NaN",
            n_bad,
        )
    out = s.where(~bad)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(out)


def apply_tcode(s: pd.Series, tcode: int) -> pd.Series:
    """Apply FRED-MD transformation ``tcode`` to series ``s``.

    ``s`` must be sorted by its natural (time) order; NaN propagates strictly
    (a NaN input at t makes every dependent output NaN, never filled).
    """
    if tcode == 1:
        return s.astype(float)
    if tcode == 2:
        return s.diff()
    if tcode == 3:
        return s.diff().diff()
    if tcode == 4:
        return _safe_log(s)
    if tcode == 5:
        return _safe_log(s).diff()
    if tcode == 6:
        return _safe_log(s).diff().diff()
    if tcode == 7:
        ratio = s / s.shift(1) - 1.0
        return ratio.diff()
    raise ValueError(f"unknown tcode: {tcode!r}")


def transform_panel(
    df: pd.DataFrame, spec: dict, allow_untransformed: bool = False
) -> pd.DataFrame:
    """Apply each column's tcode from the data config spec.

    ``spec`` is the parsed ``configs/data.yaml`` dict: ``spec['series']`` is a
    list of ``{fred_id, tcode, ...}`` entries.

    MIXED FREQUENCY (critical). A real-time panel from
    ``info_set.information_set`` is indexed by the UNION of every series'
    observation dates, so a monthly series sits on an index dominated by
    daily rows and is NaN on almost all of them. ``apply_tcode`` differences
    POSITIONALLY (``s - s.shift(1)``), so applying it to such a column
    differences against a NaN neighbour and silently annihilates the series
    (measured: 66% of PAYEMS, 74% of UNRATE, 94% of ICSA destroyed).

    Each column is therefore transformed on its OWN observed index and then
    reindexed back onto the panel, i.e. the difference is always taken
    between consecutive observations OF THAT SERIES. Reindexing introduces
    no values, so the ragged edge (R3) is preserved.

    Columns absent from the spec (e.g. the target) would otherwise pass
    through as RAW LEVELS mixed in with transformed regressors, which is a
    silent correctness trap; they raise unless ``allow_untransformed``.
    """
    tcodes = {entry["fred_id"]: entry["tcode"] for entry in spec.get("series", [])}
    unknown = [c for c in df.columns if c not in tcodes]
    if unknown and not allow_untransformed:
        raise ValueError(
            f"no tcode in spec for column(s) {unknown}; they would pass through "
            "as untransformed levels. Add them to configs/data.yaml or pass "
            "allow_untransformed=True deliberately."
        )

    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col not in tcodes:
            logger.warning("column %s passed through untransformed (raw levels)", col)
            out[col] = df[col]
            continue
        observed = df[col].dropna()
        transformed = apply_tcode(observed, tcodes[col])
        out[col] = transformed.reindex(df.index)
    return out
