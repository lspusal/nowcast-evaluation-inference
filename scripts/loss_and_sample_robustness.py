"""Is the leverage result a property of the design, or of squared error and one quarter?

The headline reductions --- 1,403 forecast pairs to 82 target periods to an
effective 44, and a variance carried by one or two quarters --- are computed
on squared-error differentials over a sample containing 2020Q2. Two objections
follow, and both are answerable rather than arguable.

* Squared error gives a quarter with a 30-point miss roughly nine hundred
  times the weight of a one-point miss. Under absolute error that factor is
  thirty. If the concentration is an artefact of the loss function it should
  largely disappear.
* 2020Q2 is a single observation unlike anything else in the record. If the
  concentration is that one quarter it should disappear when it is removed.

This recomputes the three quantities under four cells: squared and absolute
loss, with and without 2020Q2. The expected finding is that the first two
reductions are design properties and survive, while the concentration is
loss-dependent (the design notes).

Usage:  PYTHONPATH=. python scripts/loss_and_sample_robustness.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval import metrics as M

EXCLUDED_QUARTER = "2020-04-01"  # 2020Q2


def _bartlett_ess(dq: np.ndarray, lags: int | None = None) -> float:
    """G / (1 + 2 sum w_j rho_j), the variance-inflation correction."""
    g = len(dq)
    if lags is None:
        lags = max(int(np.floor(4 * (g / 100.0) ** (2.0 / 9.0))), 1)
    x = dq - dq.mean()
    denom = float((x**2).sum())
    if denom <= 0:
        return float("nan")
    infl = 1.0
    for j in range(1, lags + 1):
        rho = float((x[j:] * x[:-j]).sum() / denom)
        infl += 2.0 * (1.0 - j / (lags + 1)) * rho
    return g / infl if infl > 0 else float("nan")


def _nu(dq: np.ndarray) -> float:
    """Reciprocal Herfindahl of the leverage shares."""
    x = dq - dq.mean()
    s2 = x**2
    tot = float(s2.sum())
    if tot <= 0:
        return float("nan")
    w = s2 / tot
    return float(1.0 / (w**2).sum())


def robustness(results_dir: str = "results") -> pd.DataFrame:
    preds = M.load_predictions(results_dir)
    preds = preds[~preds["variant"].astype(str).str.contains("balanced")]
    prepared = M.prepare(preds, M.load_target())
    answered = prepared[prepared["yhat"].notna()]
    wide = answered.pivot_table(
        index=["nowcast_date", "target_quarter"], columns="model", values="error"
    )

    # The study's own headline pairs, as the effective-sample table defines them.
    ess = pd.read_csv(Path(results_dir) / "tables" / "T16_effective_sample.csv")
    pairs = list(zip(ess["model"], ess["benchmark"]))

    rows = []
    for loss in ("squared", "absolute"):
        for drop in (False, True):
            for model, bench in pairs:
                if model not in wide.columns or bench not in wide.columns:
                    continue
                pair = wide[[model, bench]].dropna()
                tq = pd.to_datetime(pair.index.get_level_values("target_quarter"))
                if drop:
                    pair = pair[tq != pd.Timestamp(EXCLUDED_QUARTER)]
                    tq = pd.to_datetime(pair.index.get_level_values("target_quarter"))
                if pair.empty:
                    continue
                if loss == "squared":
                    d = (pair[model] ** 2 - pair[bench] ** 2).to_numpy(float)
                else:
                    d = (pair[model].abs() - pair[bench].abs()).to_numpy(float)
                groups = pd.factorize(tq)[0]
                dq = np.bincount(groups, weights=d) / np.bincount(groups)
                g = len(dq)
                s, q = np.abs(dq).sum(), float((dq**2).sum())
                denom = g * q - s**2
                t_max = s * np.sqrt(g - 1) / np.sqrt(denom) if denom > 0 else np.inf
                rows.append(
                    {
                        "loss": loss,
                        "sample": "without 2020Q2" if drop else "full",
                        "model": model,
                        "benchmark": bench,
                        "n_pairs": len(pair),
                        "n_quarters": g,
                        "ess_serial": _bartlett_ess(dq),
                        "nu": _nu(dq),
                        "t_max": t_max,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()

    df = robustness(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "T22_loss_sample_robustness.csv", index=False)
    print(f"wrote {out / 'T22_loss_sample_robustness.csv'}")

    # Range of each statistic over the eleven model-benchmark pairs, per loss
    # and sample: the summary the paper tabulates. "empty_region" counts pairs
    # whose largest randomization t statistic is below the Gaussian 5% critical
    # value, so that no rejection region remains.
    g = df.groupby(["loss", "sample"], sort=False)
    summ = g.agg(G=("n_quarters", "max"),
                 ess_lo=("ess_serial", "min"), ess_hi=("ess_serial", "max"),
                 nu_lo=("nu", "min"), nu_hi=("nu", "max"),
                 tmax_lo=("t_max", "min"), tmax_hi=("t_max", "max"),
                 empty_region=("t_max", lambda x: int((x < 1.96).sum())),
                 n=("model", "size")).reset_index()
    summ["G"] = summ["G"].astype(float)
    summ.to_csv(out / "T23_robustness_summary.csv", index=False)
    print(f"wrote {out / 'T23_robustness_summary.csv'}")

    summary = (
        df.groupby(["loss", "sample"])
        .agg(
            n_pairs=("n_pairs", "median"),
            quarters=("n_quarters", "median"),
            ess_lo=("ess_serial", "min"),
            ess_hi=("ess_serial", "max"),
            nu_lo=("nu", "min"),
            nu_hi=("nu", "max"),
            tmax_lo=("t_max", "min"),
            tmax_hi=("t_max", "max"),
            cannot_reject=("t_max", lambda x: int((x < 1.99).sum())),
        )
        .round(2)
    )
    pd.set_option("display.width", 220)
    print(summary.to_string())


if __name__ == "__main__":
    main()
