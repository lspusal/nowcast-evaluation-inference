"""One table for the application, carrying effect size beside the p-values.

The application was spread over three tables that overlapped: a bandwidth
sweep, the procedures that assume independent target periods, and those that
admit dependence. A reader comparing a model across them had to hold three
column orders in mind, and none of the three carried an effect size, which is
awkward in a study whose argument is that significance and informativeness are
different things (the design notes).

This builds the single table the study reports. The cluster-robust variants,
the signed-rank test and the bandwidth sweep move to the supplement, where
they remain available as the audit trail they are.

Usage:  PYTHONPATH=. python scripts/application_table.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval import metrics as M

BOOT_DRAWS = 1999


def application(results_dir: str = "results") -> pd.DataFrame:
    tab = Path(results_dir) / "tables"
    audit = pd.read_csv(tab / "T13_procedure_audit.csv")
    ess = pd.read_csv(tab / "T16_effective_sample.csv")[
        ["model", "benchmark", "ess_serial", "t_max_randomization"]
    ]

    # The quarters-ahead count is computed here for every pair, rather than
    # merged from the versus-benchmark table, which only covers the pairs
    # whose benchmark is the expanding mean.
    preds = M.load_predictions(results_dir)
    preds = preds[~preds["variant"].astype(str).str.contains("balanced")]
    prepared = M.prepare(preds, M.load_target())
    answered = prepared[prepared["yhat"].notna()]
    wide = answered.pivot_table(
        index=["nowcast_date", "target_quarter"], columns="model", values="error"
    )

    ahead = []
    for _, r in audit.iterrows():
        a, b = r["model"], r["benchmark"]
        if a not in wide.columns or b not in wide.columns:
            ahead.append(np.nan)
            continue
        pair = wide[[a, b]].dropna()
        groups = pd.factorize(pair.index.get_level_values("target_quarter"))[0]
        d = (pair[a] ** 2 - pair[b] ** 2).to_numpy(float)
        dq = np.bincount(groups, weights=d) / np.bincount(groups)
        ahead.append(100.0 * float((dq < 0).mean()))

    out = audit.merge(ess, on=["model", "benchmark"], how="left")
    out["rel_rmse"] = out["rmse_model"] / out["rmse_benchmark"]
    out["pct_ahead"] = ahead
    # The bootstrap cannot resolve below 1/(B+1); printing 0.000 claims a
    # precision the procedure does not have.
    floor = 1.0 / (BOOT_DRAWS + 1)
    out["p_wild_bootstrap"] = out["p_wild_bootstrap"].clip(lower=floor)
    return out[
        [
            "model",
            "benchmark",
            "rel_rmse",
            "pct_ahead",
            "n_clusters",
            "ess_serial",
            "p_weekly_hac",
            "p_wild_bootstrap",
            "p_sign",
            "p_ibragimov_muller",
            "p_hac_fixed_b",
        ]
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    df = application(args.results)
    out = Path(args.results) / "tables" / "T24_application.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    pd.set_option("display.width", 220)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
