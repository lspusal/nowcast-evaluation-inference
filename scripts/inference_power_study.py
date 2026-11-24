"""Size-adjusted power of the calibrated procedures, and a sweep over the
intra-cluster correlation.

The size study (scripts/inference_size_study.py) says which procedures hold
their nominal level. It does not say what they cost, and the two we end up
recommending discard information by construction, so "their cost is power" has
to be measured rather than asserted.

Two experiments:

* ``power``  -- under the serially dependent process at the study's own grid
  density, shift the per-target effect by a mean ``mu`` and record the
  rejection rate of each procedure. Power is reported both raw and
  SIZE-ADJUSTED: a procedure that over-rejects under the null would otherwise
  look powerful for the wrong reason, and the whole point of the size study is
  that several of these do. The size-adjusted critical value is the empirical
  alpha-quantile of that procedure's null p-value distribution, taken from the
  mu = 0 row of the same simulation.
* ``icc``    -- vary tau, which sets the intra-cluster correlation
  ICC = tau^2 / (tau^2 + 1), and record size at the study's overlap. This
  answers what the headline distortion depends on, since the 28.9% figure is a
  function of the ICC and not of the grid alone.

Usage:  PYTHONPATH=. python scripts/inference_power_study.py --reps 2000
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.cluster_inference import (
    cluster_robust,
    hac_quarterly,
    hac_quarterly_fixed_b,
    ibragimov_muller,
    moving_block_bootstrap,
    wild_cluster_bootstrap,
)
from src.eval.dm_test import _newey_west_variance

NOMINAL = 0.05
BOOT = 199


def _p_weekly(d, g, n):
    """The weekly test as this project ran it: Newey-West at bandwidth 4 on
    the stacked series, with the Harvey-Leybourne-Newbold factor at h = 1.

    It takes the loss DIFFERENTIAL directly. Handing `dm_test` the
    differential and a vector of zeros instead compares a series against
    zero, whose differential has a positive mean by construction and which
    therefore rejects always.
    """
    from scipy import stats

    lrv = _newey_west_variance(d, 4)
    if not np.isfinite(lrv) or lrv <= 0:
        return 1.0
    stat = d.mean() / np.sqrt(lrv / len(d))
    stat *= np.sqrt((len(d) - 1) / len(d))
    return float(2.0 * (1.0 - stats.t.cdf(abs(stat), df=len(d) - 1)))


def _p_quarterly_t(d, g, n):
    from scipy import stats

    dq = np.bincount(g, weights=d) / np.bincount(g)
    if dq.std(ddof=1) <= 0:
        return np.nan
    t = dq.mean() / (dq.std(ddof=1) / np.sqrt(len(dq)))
    return float(2 * (1 - stats.t.cdf(abs(t), df=len(dq) - 1)))


def _p_sign(d, g, n):
    from scipy import stats

    dq = np.bincount(g, weights=d) / np.bincount(g)
    k = int((dq < 0).sum())
    m = int((dq != 0).sum())
    return float(stats.binomtest(k, m, 0.5).pvalue) if m else np.nan


def _p_im(d, g, n, *, q):
    return ibragimov_muller(d, g, q=q)["p"]


def _p_fixed_b(d, g, n, *, b):
    return hac_quarterly_fixed_b(d, g, b=b)["p"]


# The procedures whose cost the study has to quantify: the conventional one,
# the best of the cluster-level family, the rank alternative, the two
# asymptotically dependence-robust ones, and the two recommended.
PROCEDURES = {
    "weekly_hac4": _p_weekly,
    "cluster_cr1": lambda d, g, n: cluster_robust(d, g)["p"],
    "wild_bootstrap": lambda d, g, n: wild_cluster_bootstrap(d, g, n_boot=BOOT, seed=0)[
        "p"
    ],
    "quarterly_t": _p_quarterly_t,
    "quarterly_sign": _p_sign,
    "hac_quarterly": lambda d, g, n: hac_quarterly(d, g)["p"],
    "moving_block": lambda d, g, n: moving_block_bootstrap(d, g, n_boot=BOOT, seed=0)[
        "p"
    ],
    "ibragimov_muller_q4": partial(_p_im, q=4),
    "hac_fixed_b025": partial(_p_fixed_b, b=0.25),
}


def _ar1(rng, rho: float, sigma: float, m: int) -> np.ndarray:
    """Stationary AR(1) with POPULATION mean zero."""
    e = rng.normal(0.0, sigma, m)
    out = np.empty(m)
    out[0] = e[0] / np.sqrt(1 - rho**2)
    for t in range(1, m):
        out[t] = rho * out[t - 1] + e[t]
    return out


def _pvalues(draw, n_reps: int, qidx: np.ndarray, n_q: int) -> pd.DataFrame:
    rows = []
    for _ in range(n_reps):
        d = draw()
        rows.append({k: fn(d, qidx, n_q) for k, fn in PROCEDURES.items()})
    return pd.DataFrame(rows)


def power_experiment(n_reps: int, seed: int, heavy_tail: bool = False) -> pd.DataFrame:
    """Size-adjusted power under the serially dependent process.

    With ``heavy_tail``, one quarter's effect is inflated to eight standard
    deviations before the mean shift is applied, reproducing the leverage of
    2020Q2 exactly as the size study's heavy-tail process does. the study
    argues that leverage is what empties the quarter-aggregated t-test's
    rejection region, so a power comparison run only on Gaussian effects would
    be measuring the case the study says does not obtain.
    """
    rng = np.random.default_rng(seed)
    n_q, per_q, tau, rho = 82, 17, 1.0, 0.4
    qidx = np.repeat(np.arange(n_q), per_q)
    n = len(qidx)
    mus = [0.0, 0.10, 0.20, 0.30, 0.40, 0.60]
    spike_at = n_q // 2

    def draw(mu):
        def _d():
            u = _ar1(rng, rho, tau, n_q)
            if heavy_tail:
                # The spike multiplies the DEVIATION, so the shift mu stays the
                # quantity being detected rather than being scaled with it.
                u[spike_at] *= 8.0
            u = u + mu
            return u[qidx] + rng.normal(0.0, 1.0, n)

        return _d

    # The null row first: its p-value quantiles are the size-adjusted
    # critical values every later row is judged against.
    null_p = _pvalues(draw(0.0), n_reps, qidx, n_q)
    crit = {c: float(np.nanquantile(null_p[c], NOMINAL)) for c in PROCEDURES}

    rows = []
    for mu in mus:
        p = null_p if mu == 0.0 else _pvalues(draw(mu), n_reps, qidx, n_q)
        row = {
            "mu": mu,
            "process": "heavy_tail" if heavy_tail else "serial",
            "n_quarters": n_q,
            "dates_per_quarter": per_q,
            "rho": rho,
            "tau": tau,
            "n_reps": n_reps,
        }
        for c in PROCEDURES:
            v = p[c].to_numpy(float)
            row[f"{c}_raw"] = float(np.nanmean(v < NOMINAL))
            row[f"{c}_adj"] = float(np.nanmean(v <= crit[c]))
            if mu == 0.0:
                row[f"{c}_crit"] = crit[c]
        rows.append(row)
    return pd.DataFrame(rows)


def icc_experiment(n_reps: int, seed: int) -> pd.DataFrame:
    """Size at the study's overlap, as a function of the intra-cluster
    correlation. tau sets ICC = tau^2 / (1 + tau^2)."""
    rng = np.random.default_rng(seed)
    n_q, per_q = 82, 17
    qidx = np.repeat(np.arange(n_q), per_q)
    n = len(qidx)
    rows = []
    for tau in (0.0, 0.3, 0.5, 0.7, 1.0, 1.71):

        def draw(tau=tau):
            def _d():
                u = rng.normal(0.0, tau, n_q) if tau > 0 else np.zeros(n_q)
                return u[qidx] + rng.normal(0.0, 1.0, n)

            return _d

        p = _pvalues(draw(), n_reps, qidx, n_q)
        icc = tau**2 / (1 + tau**2)
        rows.append(
            {
                "tau": tau,
                "icc": icc,
                "kish_deff": 1 + (per_q - 1) * icc,
                "n_quarters": n_q,
                "dates_per_quarter": per_q,
                "n_reps": n_reps,
                **{
                    c: float(np.nanmean(p[c].to_numpy(float) < NOMINAL))
                    for c in PROCEDURES
                },
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    icc = icc_experiment(args.reps, args.seed)
    icc.to_csv(out / "T18_size_by_icc.csv", index=False)
    print(f"wrote {out / 'T18_size_by_icc.csv'}")

    pw = pd.concat(
        [
            power_experiment(args.reps, args.seed + 1),
            power_experiment(args.reps, args.seed + 2, heavy_tail=True),
        ],
        ignore_index=True,
    )
    pw.to_csv(out / "T19_power.csv", index=False)
    print(f"wrote {out / 'T19_power.csv'}")

    pd.set_option("display.width", 220)
    cols = [
        "icc",
        "kish_deff",
        "weekly_hac4",
        "cluster_cr1",
        "ibragimov_muller_q4",
        "hac_fixed_b025",
    ]
    print("\n== size by ICC ==")
    print(icc[cols].round(3).to_string(index=False))
    adj = ["mu"] + [
        f"{c}_adj"
        for c in (
            "cluster_cr1",
            "wild_bootstrap",
            "quarterly_sign",
            "moving_block",
            "ibragimov_muller_q4",
            "hac_fixed_b025",
        )
    ]
    print("\n== size-adjusted power ==")
    print(pw[["process"] + adj].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
