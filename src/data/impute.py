"""Backward-only imputation for the ragged edge.

Minimal implementation of docs/02_methodology.md §3 ("imputación solo hacia
atrás: último valor disponible / media expanding, nunca futura"), kept here
as a small standalone utility so that R6's leakage tests can run at M0
against the real function.

M2 MUST import this rather than reimplement it in ml_baselines.py (whose
`backward_impute` stub should delegate here): two implementations of the
backward-only rule means two chances to leak, and only one of them would be
covered by the anti-leakage tests.
"""

from __future__ import annotations

import pandas as pd


def backward_impute(df: pd.DataFrame) -> pd.DataFrame:
    """LOCF per column; remaining leading NaNs filled with the expanding mean.

    Row order encodes time (ascending). Only values at or before a given row
    are ever used to fill that row — R1.
    """
    locf = df.ffill()
    expanding_mean = df.expanding(min_periods=1).mean()
    return locf.fillna(expanding_mean)
