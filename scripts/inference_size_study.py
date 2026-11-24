"""Empirical size of predictive-accuracy tests on the real-time nowcast design.

Produces `results/tables/T10_inference_size.csv`. This is the artifact that
turns from an erratum into a methodological result:
without it, "our HAC(4) p-values were wrong" reads as this project's mistake
rather than a property of the design every real-time nowcasting study shares.

Two independent measurements:

PART A -- synthetic, under an exact null. The loss differential is
``d_it = u_q(i) + e_it`` with ``u_q ~ N(0, tau^2)`` a per-target-quarter
effect and ``e_it ~ N(0, 1)``, mean zero by construction, Gaussian, no fat
tails. Any rejection is a size distortion attributable to the OVERLAP alone
(``dates_per_quarter`` nowcast dates aimed at the same target), so the result
does not depend on this study's data at all and a referee can reproduce it
without the cache.

PART B -- randomization on the real loss differentials. Under H0 of equal
predictive accuracy the sign of each target quarter's mean differential is
exchangeable, so flipping the sign of WHOLE quarters preserves the exact
intra-quarter dependence and tail structure while imposing the null. The
rejection rate of each asymptotic test against this null is its empirical
size on real data.

The two parts fail in OPPOSITE directions, which is the point:
* the weekly HAC test is grossly oversized, by the overlap;
* the quarterly t-test is correctly sized on Gaussian data but structurally
  unable to reject on real loss differentials, because one crisis quarter
  dominates the sample standard deviation;
* the exact sign test is correctly sized in both.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.eval.cluster_inference import (
    cluster_jackknife,
    cluster_robust,
    cluster_robust_cr2,
    hac_quarterly,
    hac_quarterly_fixed_b,
    ibragimov_muller,
    moving_block_bootstrap,
    randomization_mean,
    wild_cluster_bootstrap,
)
from src.eval.dm_test import _newey_west_variance

NOMINAL = 0.05
HAC_LAGS = (4, 8, 17, 26, 52)
# The comparisons the study's inference audit (T8) reports.
REAL_PAIRS = (
    ("chronos_completed_dfm", "dfm_robust"),
    ("dfm", "naive_mean"),
    ("dfm_robust", "naive_mean"),
    ("bridge", "dfm_robust"),
    ("xgboost", "naive_mean"),
)


def _p_weekly_hac(d: np.ndarray, lags: int) -> float:
    """The test as this project ran it: weekly n, Newey-West, HLN at h=1."""
    n = len(d)
    lrv = _newey_west_variance(d, lags)
    if not np.isfinite(lrv) or lrv <= 0:
        return 1.0
    stat = d.mean() / np.sqrt(lrv / n)
    stat *= np.sqrt((n - 1) / n)  # HLN correction factor at h = 1
    return float(2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1)))


def _p_quarterly_t(d: np.ndarray, qidx: np.ndarray, n_q: int) -> float:
    dq = np.bincount(qidx, weights=d) / np.bincount(qidx)
    sd = dq.std(ddof=1)
    if sd == 0:
        return 1.0
    t = dq.mean() / (sd / np.sqrt(n_q))
    return float(2.0 * (1.0 - stats.t.cdf(abs(t), df=n_q - 1)))


def _p_sign(d: np.ndarray, qidx: np.ndarray, n_q: int) -> float:
    dq = np.bincount(qidx, weights=d) / np.bincount(qidx)
    return float(stats.binomtest(int((dq < 0).sum()), n_q, 0.5).pvalue)


def _p_wilcoxon(d: np.ndarray, qidx: np.ndarray, n_q: int) -> float:
    dq = np.bincount(qidx, weights=d) / np.bincount(qidx)
    try:
        return float(stats.wilcoxon(dq)[1])
    except ValueError:
        return 1.0


# Resampling draws inside each replication. Deliberately small: the quantity
# being measured is a rejection RATE over many replications, and 199 draws
# already resolve a 5% threshold to about +-0.005.
WILD_DRAWS = 199


def _p_weekly_hac_at(d, g, n, *, lag: int) -> float:
    return _p_weekly_hac(d, lag)


def _p_im(d, g, n, *, q: int, rotate: float = 0.0) -> float:
    return ibragimov_muller(d, g, q=q, rotate=rotate)["p"]


def _p_fixed_b(d, g, n, *, b: float) -> float:
    return hac_quarterly_fixed_b(d, g, b=b)["p"]


# Group counts for the Ibragimov--Muller test and bandwidth fractions for the
# fixed-b HAC test. Both are swept rather than fixed because the study's
# recommendation rests on them, and a recommendation whose size depended on an
# arbitrary tuning choice would not be one.
IM_GROUPS = (2, 3, 4, 5, 6, 8)
FIXED_B = (0.1, 0.25, 0.5)

# The full panel of procedures a referee would ask about. The cluster-robust
# family is the obvious answer to "the errors are clustered by target
# period", so it is measured rather than dismissed, and the resampling
# procedures are the ones that survive on real data
#.
PROCEDURES: dict[str, object] = {
    **{f"weekly_hac{lag}": partial(_p_weekly_hac_at, lag=lag) for lag in HAC_LAGS},
    "cluster_cr1": lambda d, g, n: cluster_robust(d, g)["p"],
    "cluster_cr2": lambda d, g, n: cluster_robust_cr2(d, g)["p"],
    "cluster_cr3": lambda d, g, n: cluster_jackknife(d, g)["p"],
    "wild_bootstrap": lambda d, g, n: wild_cluster_bootstrap(
        d, g, n_boot=WILD_DRAWS, seed=0
    )["p"],
    "randomization_mean": lambda d, g, n: randomization_mean(
        d, g, n_draws=WILD_DRAWS, seed=0
    )["p"],
    "quarterly_t": _p_quarterly_t,
    "quarterly_wilcoxon": _p_wilcoxon,
    "quarterly_sign": _p_sign,
    # Valid under weak DEPENDENCE between clusters, unlike everything above.
    "hac_quarterly": lambda d, g, n: hac_quarterly(d, g)["p"],
    "moving_block": lambda d, g, n: moving_block_bootstrap(
        d, g, n_boot=WILD_DRAWS, seed=0
    )["p"],
    # Built for FEW, heterogeneous, temporally dependent groups rather than
    # for large G -- the regime the other ten are asked to work in and do not.
    # Both are expected to be conservative; the question the study asks is
    # whether either one is correctly sized where the rest are liberal.
    **{f"ibragimov_muller_q{q}": partial(_p_im, q=q) for q in IM_GROUPS},
    # Same procedure, groups rotated by half a group, to check the verdict is
    # not an artifact of where the partition happens to fall.
    "ibragimov_muller_q4_rot": partial(_p_im, q=4, rotate=0.5),
    **{
        f"hac_fixed_b{str(bw).replace('.', '')}": partial(_p_fixed_b, b=bw)
        for bw in FIXED_B
    },
}


# The single source of truth for "how many procedures does this study
# measure". A family is one procedure; several columns of the same family are
# the same procedure at different settings (five Newey--West bandwidths, two
# group counts for Ibragimov--Muller). The study's counts are derived from
# this map by tests/test_paper_claims.py rather than typed, because they were
# typed once and went stale when the panel grew.
PROCEDURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "weekly Diebold--Mariano": tuple(f"weekly_hac{lag}" for lag in HAC_LAGS),
    "cluster-robust CR1": ("cluster_cr1",),
    "cluster-robust CR2": ("cluster_cr2",),
    "cluster-robust CR3": ("cluster_cr3",),
    "wild cluster bootstrap": ("wild_bootstrap",),
    "randomization on the cluster mean": ("randomization_mean",),
    "quarter-aggregated t": ("quarterly_t",),
    "quarter-aggregated signed-rank": ("quarterly_wilcoxon",),
    "quarter-aggregated sign": ("quarterly_sign",),
    "HAC on cluster means": ("hac_quarterly",),
    "moving-block bootstrap": ("moving_block",),
    "Ibragimov--Muller": tuple(f"ibragimov_muller_q{q}" for q in IM_GROUPS)
    + ("ibragimov_muller_q4_rot",),
    "fixed-b HAC": tuple(f"hac_fixed_b{str(bw).replace('.', '')}" for bw in FIXED_B),
}


# The quarter-level log variance ratio for the squared-loss processes, and the
# shift that makes E[exp(l) - exp(-l)] = 0 so the null is exactly equal mean
# loss. For a symmetric l the shift is zero by symmetry; for a skewed one it
# is estimated once from a large draw under a fixed seed, so the processes
# stay reproducible.
_RATIO_SPEC = {
    "squared_loss": ("normal", 1.2),
    "squared_skewed": ("gumbel", 0.6),
    "squared_mixture": ("mixture", 1.2),
    # The combination the real differentials actually show: an equal-mean-loss
    # null built from squared errors, a skewed quarter effect, and persistence
    # between quarters. Every other process isolates one of these, so the
    # recommendation rested on mechanisms tested one at a time.
    "squared_skewed_serial": ("gumbel", 0.6),
}
# Processes whose quarter effect is persistent, with the first-order
# autocorrelation we measure on our own differentials.
_RATIO_RHO = {"squared_skewed_serial": 0.4}
_RATIO_SHIFT: dict[str, float] = {}


def _ratio_draw(rng, kind: str, scale: float, size: int) -> np.ndarray:
    if kind == "normal":
        return rng.normal(0.0, scale, size)
    if kind == "gumbel":
        return rng.gumbel(0.0, scale, size)
    heavy = rng.random(size) < 0.1
    return np.where(
        heavy,
        rng.normal(2.0 * scale, 0.5 * scale, size),
        rng.normal(-0.2 * scale, 0.3 * scale, size),
    )


def _ratio_path(rng, kind: str, scale: float, size: int, rho: float) -> np.ndarray:
    """The quarter-level log variance ratio, persistent when ``rho`` is set.

    The AR(1) is driven by the same skewed innovation, so the marginal stays
    asymmetric while successive quarters are correlated. That marginal is not
    the innovation's, which is why the centring constant is calibrated on the
    path rather than reused from the independent case.
    """
    e = _ratio_draw(rng, kind, scale, size)
    if not rho:
        return e
    out = np.empty(size)
    out[0] = e[0] / np.sqrt(1 - rho**2)
    for t in range(1, size):
        out[t] = rho * out[t - 1] + e[t]
    return out


def _ratio_effect(rng, dgp: str, n_q: int) -> np.ndarray:
    kind, scale = _RATIO_SPEC[dgp]
    rho = _RATIO_RHO.get(dgp, 0.0)
    if dgp not in _RATIO_SHIFT:
        if kind == "normal" and not rho:
            _RATIO_SHIFT[dgp] = 0.0  # symmetric: E[e^l] = E[e^-l]
        else:
            # E[d] = 0 has to hold for the marginal actually simulated, so
            # the shift is measured on a long path of the same process. The
            # sums are accumulated in blocks: exp(l) has a heavy right tail,
            # and at five million draws the shift was imprecise enough to
            # leave the simulated null a couple of standard errors off zero.
            cal = np.random.default_rng(20260101)
            tot_pos = tot_neg = 0.0
            n_cal = 0
            for _ in range(40):
                blk = _ratio_path(cal, kind, scale, 2_500_000, rho)
                tot_pos += float(np.exp(blk).sum())
                tot_neg += float(np.exp(-blk).sum())
                n_cal += blk.size
            _RATIO_SHIFT[dgp] = float(np.log(tot_pos / tot_neg) / 2.0)
    return _ratio_path(rng, kind, scale, n_q, rho) - _RATIO_SHIFT[dgp]


def _rejection_rates(draw, n_reps: int, qidx: np.ndarray, n_q: int) -> dict[str, float]:
    counts = dict.fromkeys(PROCEDURES, 0)
    for _ in range(n_reps):
        d = draw()
        for name, fn in PROCEDURES.items():
            p = fn(d, qidx, n_q)
            if np.isfinite(p) and p < NOMINAL:
                counts[name] += 1
    return {k: v / n_reps for k, v in counts.items()}


def part_a(n_reps: int, seed: int) -> pd.DataFrame:
    """Synthetic nulls, eleven data-generating processes.

    The first isolates the mechanical effect of the overlap and is the
    cleanest demonstration; the rest add, one at a time, features of real
    macroeconomic loss differentials. The point is not universality but
    attribution: each row says which feature a given procedure's failure is
    caused by.

    The last four exist specifically to stress the two procedures the study
    ends up recommending, since a recommendation supported by one AR(1) at
    one persistence would not be one. They ask
    what happens when the dependence is stronger than measured, when it is
    long-range rather than short-range, when it is a regime shift rather than
    a stationary AR process, and when there are half as many clusters.

    * ``overlap``      -- Gaussian random intercept, balanced grid. The pure
                          within-target overlap.
    * ``serial``       -- the per-target effect follows an AR(1) with
                          rho = 0.4, matching the autocorrelation measured on
                          our own quarterly differentials.
    * ``heavy_tail``   -- one target's effect is inflated to eight standard
                          deviations, reproducing the leverage of 2020Q2.
    * ``unbalanced``   -- cluster sizes drawn from 1 to 18, matching the
                          spread the backcast and forecast edges create.
    * ``serial_strong``-- AR(1) at rho = 0.7, well beyond the +0.48 maximum
                          we measure, to see where the recommendation breaks.
    * ``long_memory``  -- AR(1) at rho = 0.4 plus an AR(1) at rho = 0.95 with
                          half its innovation scale. The near-unit-root
                          component carries 68% of the variance of the
                          per-target effect and takes its first-order
                          autocorrelation from 0.40 to 0.78, with 0.56 still
                          left at lag four. This is strong persistence, NOT
                          long memory in the ARFIMA sense: the
                          autocorrelations are summable. the study calls it
                          "near unit root" for that reason.
    * ``regime``       -- no AR process at all. Two halves with different
                          variance (1 and 3) and different persistence (0.1
                          and 0.6), mean zero throughout: dependence as a
                          structural break rather than as stationary decay.
    * ``few_clusters`` -- the serial process at G = 40, since both
                          recommended procedures are justified for small G
                          and that claim should be tested downward.
    """
    rng = np.random.default_rng(seed)
    rows = []
    grid = [
        ("overlap", 82, 1, 1.0),
        ("overlap", 82, 4, 1.0),
        ("overlap", 82, 17, 1.0),
        ("overlap", 82, 34, 1.0),
        ("serial", 82, 17, 1.0),
        ("heavy_tail", 82, 17, 1.0),
        ("unbalanced", 82, 17, 1.0),
        ("serial_strong", 82, 17, 1.0),
        ("long_memory", 82, 17, 1.0),
        ("regime", 82, 17, 1.0),
        ("few_clusters", 40, 17, 1.0),
        ("squared_loss", 82, 17, 1.0),
        ("squared_skewed", 82, 17, 1.0),
        ("squared_mixture", 82, 17, 1.0),
        ("squared_skewed_serial", 82, 17, 1.0),
    ]
    for dgp, n_q, per_q, tau in grid:
        if dgp == "unbalanced":
            sizes = rng.integers(1, 19, size=n_q)
            qidx = np.repeat(np.arange(n_q), sizes)
        else:
            qidx = np.repeat(np.arange(n_q), per_q)
        n = len(qidx)

        def _ar1(rho: float, sigma: float, m: int) -> np.ndarray:
            """A stationary AR(1) with POPULATION mean zero.

            Not sample mean zero: centring the draw would remove the
            between-target variance from the numerator while leaving it in
            the denominator, and no test could then reject.
            """
            e = rng.normal(0.0, sigma, m)
            out = np.empty(m)
            out[0] = e[0] / np.sqrt(1 - rho**2)
            for t in range(1, m):
                out[t] = rho * out[t - 1] + e[t]
            return out

        def draw(dgp=dgp, n_q=n_q, tau=tau, qidx=qidx, n=n, _ar1=_ar1):
            if dgp in ("serial", "few_clusters"):
                u = _ar1(0.4, tau, n_q)
            elif dgp == "serial_strong":
                u = _ar1(0.7, tau, n_q)
            elif dgp == "long_memory":
                # The near-unit-root component carries 68% of the variance
                # of u and puts most of the long-run variance beyond any
                # bandwidth a plug-in rule would pick.
                u = _ar1(0.4, tau, n_q) + 0.5 * _ar1(0.95, tau, n_q)
            elif dgp == "regime":
                half = n_q // 2
                u = np.concatenate(
                    [_ar1(0.1, tau, half), _ar1(0.6, 3.0 * tau, n_q - half)]
                )
            elif dgp.startswith("squared_"):
                # Everything above draws the DIFFERENTIAL directly, which is
                # symmetric by construction, so the sign-flip null holds in
                # the simulation as well as the equal-mean null. These three
                # impose only E[d] = 0, which is what a Diebold-Mariano test
                # targets, by building d from genuine squared errors.
                #
                # The mechanism has to be the one real data has: a quarter in
                # which model A does relatively better and another in which B
                # does, averaging out. A shared difficulty term does NOT do
                # it --- it cancels in e_A^2 - e_B^2 and leaves an
                # intra-cluster correlation near zero, removing the very
                # pseudoreplication the study is about. What works is a
                # quarter-level log variance RATIO: model A has the larger
                # error variance in some quarters and the smaller in others.
                ell = _ratio_effect(rng, dgp, n_q)
                s_a = np.exp(ell / 2.0)[qidx]
                s_b = np.exp(-ell / 2.0)[qidx]
                e_a = s_a * rng.normal(0.0, 1.0, n)
                e_b = s_b * rng.normal(0.0, 1.0, n)
                return e_a**2 - e_b**2
            else:
                u = rng.normal(0.0, tau, n_q)
                if dgp == "heavy_tail":
                    u[rng.integers(0, n_q)] *= 8.0
            return u[qidx] + rng.normal(0.0, 1.0, n)

        rates = _rejection_rates(draw, n_reps, qidx, n_q)
        rows.append(
            {
                "part": "A_synthetic_null",
                "label": f"A ({dgp})",
                "comparison": dgp,
                "n_quarters": n_q,
                "dates_per_quarter": round(n / n_q, 1),
                "n_pairs": n,
                "excess_kurtosis": 0.0,
                **rates,
            }
        )
    return pd.DataFrame(rows)


def part_b(n_reps: int, seed: int, results_dir: str) -> pd.DataFrame:
    """Randomization null on the real loss differentials."""
    from src.eval import metrics as M

    preds = M.load_predictions(results_dir)
    if preds.empty:
        return pd.DataFrame()
    preds = preds[~preds["variant"].astype(str).str.contains("balanced")]
    prep = M.prepare(preds, M.load_target())
    answered = prep[prep["yhat"].notna()]
    wide = answered.pivot_table(
        index=["nowcast_date", "target_quarter"], columns="model", values="error"
    )

    rng = np.random.default_rng(seed)
    rows = []
    for model, bench in REAL_PAIRS:
        if model not in wide.columns or bench not in wide.columns:
            continue
        pair = wide[[model, bench]].dropna()
        if len(pair) < 100:
            continue
        d = (pair[model] ** 2 - pair[bench] ** 2).to_numpy(float)
        qidx = pd.factorize(pair.index.get_level_values("target_quarter"))[0]
        n_q = int(qidx.max()) + 1
        dq = np.bincount(qidx, weights=d) / np.bincount(qidx)

        def draw(d=d, qidx=qidx, n_q=n_q):
            # One sign per TARGET QUARTER: preserves the exact intra-quarter
            # dependence and the tail structure while imposing the null.
            return d * rng.choice([-1.0, 1.0], size=n_q)[qidx]

        rates = _rejection_rates(draw, n_reps, qidx, n_q)
        rows.append(
            {
                "part": "B_randomization_on_real_data",
                "label": "B (real data)",
                "comparison": f"{model} vs {bench}",
                "n_quarters": n_q,
                "dates_per_quarter": round(len(d) / n_q, 1),
                "n_pairs": len(d),
                "excess_kurtosis": float(stats.kurtosis(dq)),
                **rates,
            }
        )
    return pd.DataFrame(rows)


# Figures for these artifacts are built by src/viz/figures.py against a
# validated design system; this script writes
# only the table, so a figure has exactly one owner.


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/tables/T10_inference_size.csv")
    args = ap.parse_args()

    frames = [
        part_a(args.reps, args.seed),
        part_b(args.reps, args.seed + 1, args.results_dir),
    ]
    table = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    table["nominal_size"] = NOMINAL
    table["n_reps"] = args.reps
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    cols = [c for c in table.columns if c in PROCEDURES]
    print(f"wrote {args.out}  ({args.reps} reps)\n")
    show = table[["part", "comparison", "dates_per_quarter", *cols]]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
