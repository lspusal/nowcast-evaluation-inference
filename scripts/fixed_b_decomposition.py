"""What does the fixed-b reading actually change: the bandwidth, or the reference?

`hac_quarterly_fixed_b` differs from `hac_quarterly` in two ways at once. It
widens the Bartlett bandwidth from the automatic L = 3 at G = 82 to L = bG,
which is 20 at b = 0.25, and it replaces the t reference distribution with the
simulated fixed-b one. A referee is entitled to ask which of the two repairs
the size, and the size study cannot say because it only reports the endpoints.

This runs the intermediate case. Four readings of the SAME sequence of cluster
means under the serially dependent process of the size study:

* ``t_plain``        -- the one-sample t-test, no HAC correction at all
* ``hac_auto_t``     -- HAC at the automatic bandwidth, against t(G-1)
* ``hac_wide_t``     -- HAC at the fixed-b bandwidth, still against t(G-1)
* ``hac_wide_fixedb``-- the same statistic against the fixed-b critical value

The answer is not the one the framing invites: the wider bandwidth on its own
makes the test WORSE, and the reference distribution does all of the repair.

Usage:  PYTHONPATH=. python scripts/fixed_b_decomposition.py --reps 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.cluster_inference import _fixed_b_critical, hac_quarterly

NOMINAL = 0.05


def _ar1(rng, rho: float, sigma: float, m: int) -> np.ndarray:
    """Stationary AR(1) with POPULATION mean zero."""
    e = rng.normal(0.0, sigma, m)
    out = np.empty(m)
    out[0] = e[0] / np.sqrt(1 - rho**2)
    for t in range(1, m):
        out[t] = rho * out[t - 1] + e[t]
    return out


def decomposition(n_reps: int, seed: int, b: float = 0.25) -> pd.DataFrame:
    n_q, per_q, rho, tau = 82, 17, 0.4, 1.0
    l_auto = int(np.floor(4 * (n_q / 100.0) ** (2.0 / 9.0)))
    l_wide = max(round(b * n_q), 1)
    crit_t = float(stats.t.ppf(1 - NOMINAL / 2, n_q - 1))
    crit_b = _fixed_b_critical(b)

    rng = np.random.default_rng(seed)
    qidx = np.repeat(np.arange(n_q), per_q)
    n = len(qidx)
    hits = dict.fromkeys(["t_plain", "hac_auto_t", "hac_wide_t", "hac_wide_fixedb"], 0)

    for _ in range(n_reps):
        d = _ar1(rng, rho, tau, n_q)[qidx] + rng.normal(0.0, 1.0, n)
        dq = np.bincount(qidx, weights=d) / np.bincount(qidx)
        hits["t_plain"] += abs(dq.mean() / (dq.std(ddof=1) / np.sqrt(n_q))) > crit_t
        hits["hac_auto_t"] += abs(hac_quarterly(d, qidx, lags=l_auto)["stat"]) > crit_t
        wide = abs(hac_quarterly(d, qidx, lags=l_wide)["stat"])
        hits["hac_wide_t"] += wide > crit_t
        hits["hac_wide_fixedb"] += wide > crit_b

    rows = []
    for name, k in hits.items():
        r = k / n_reps
        rows.append(
            {
                "reading": name,
                # The plain t applies no HAC correction at all, so it has no
                # bandwidth; NaN prints as an em dash rather than as a zero
                # a reader would take for a bandwidth of zero lags.
                "bandwidth": {
                    "t_plain": float("nan"),
                    "hac_auto_t": l_auto,
                    "hac_wide_t": l_wide,
                    "hac_wide_fixedb": l_wide,
                }[name],
                "critical_value": crit_b if name.endswith("fixedb") else crit_t,
                "size_pct": 100 * r,
                "mc_se_pct": 100 * np.sqrt(r * (1 - r) / n_reps),
                "n_quarters": n_q,
                "dates_per_quarter": per_q,
                "rho": rho,
                "b": b,
                "n_reps": n_reps,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--b", type=float, default=0.25)
    ap.add_argument("--out", default="results/tables")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = decomposition(args.reps, args.seed, args.b)
    df.to_csv(out / "T20_fixed_b_decomposition.csv", index=False)
    print(f"wrote {out / 'T20_fixed_b_decomposition.csv'}")
    pd.set_option("display.width", 200)
    print(
        df[["reading", "bandwidth", "critical_value", "size_pct", "mc_se_pct"]]
        .round(2)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
