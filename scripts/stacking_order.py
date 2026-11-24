"""Does the weekly test's verdict depend on how the panel is flattened?

A nowcast evaluation is two-dimensional: a target quarter and a date at which
it is nowcast. A Diebold-Mariano test wants one sequence, so the panel has to
be flattened, and a HAC variance depends on the order chosen. We stack by
target quarter and then by nowcast date within it, the order in which a
practitioner accumulates the series. Calendar order is equally natural: on any
Monday a system prints a backcast, a nowcast and a forecast, and the next
Monday it prints three more.

If the over-rejection we report survives both orders it is a property of the
design. If it does not, flattening is itself an inferential choice and should
be reported as one.

Writes results/tables/T33_stacking_order.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.eval import metrics
from src.eval.dm_test import dm_test

AUDIT = "results/tables/T13_procedure_audit.csv"
HAC_LAGS = 4


def _series(df: pd.DataFrame, a: str, b: str, order: str) -> pd.DataFrame:
    x = df[df.model == a][["nowcast_date", "target_quarter", "error"]]
    y = df[df.model == b][["nowcast_date", "target_quarter", "error"]]
    m = x.merge(y, on=["nowcast_date", "target_quarter"], suffixes=("_a", "_b"))
    m = m.dropna(subset=["error_a", "error_b"])
    if order == "target_first":
        m = m.sort_values(["target_quarter", "nowcast_date"])
    elif order == "calendar":
        m = m.sort_values(["nowcast_date", "target_quarter"])
    elif order == "horizon_first":
        m = m.assign(h=(m.target_quarter - m.nowcast_date).dt.days)
        m = m.sort_values(["h", "nowcast_date"])
    else:
        raise ValueError(order)
    return m.reset_index(drop=True)


def main() -> None:
    audit = pd.read_csv(AUDIT)
    df = metrics.prepare(metrics.load_predictions("results"), metrics.load_target())
    rows = []
    for _, r in audit.iterrows():
        a, b = r["model"], r["benchmark"]
        if a not in set(df.model) or b not in set(df.model):
            continue
        row = {"model": a, "benchmark": b}
        for order in ("target_first", "calendar", "horizon_first"):
            m = _series(df, a, b, order)
            _, p = dm_test(m["error_a"], m["error_b"], hac_lags=HAC_LAGS)
            row[f"p_{order}"] = float(p)
            d = (m.error_a**2 - m.error_b**2).to_numpy(float)
            d = d - d.mean()
            row[f"rho1_{order}"] = float(np.corrcoef(d[:-1], d[1:])[0, 1])
        row["n"] = len(m)
        rows.append(row)

    out = pd.DataFrame(rows)
    path = Path("results/tables/T33_stacking_order.csv")
    out.to_csv(path, index=False)
    print(f"wrote {path}")
    cols = [
        "model",
        "benchmark",
        "n",
        "p_target_first",
        "p_calendar",
        "p_horizon_first",
    ]
    print(out[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    for order in ("target_first", "calendar", "horizon_first"):
        p = out[f"p_{order}"]
        rho = out[f"rho1_{order}"]
        print(
            f"{order:<14s} significant at 5%: {int((p < 0.05).sum()):2d} of "
            f"{len(p)}   first-order autocorrelation "
            f"{rho.min():.2f} to {rho.max():.2f}"
        )


if __name__ == "__main__":
    main()
