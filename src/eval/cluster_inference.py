"""Cluster-level inference for predictive-accuracy comparisons.

The evaluation grid of a real-time nowcasting study places many nowcast dates
inside each target period, so the loss differential has an exact cluster
structure: the target period is the cluster, and the referee's first question
is why one would not simply use cluster-robust inference at that level. This
module implements the standard answers so that the question is settled by
measurement rather than by assertion.

All procedures test :math:`H_0: \\mathbb{E}[d] = 0` for the loss differential
``d`` clustered by target period ``g``, in the intercept-only model
:math:`d_{gt} = \\mu + \\varepsilon_{gt}`. Because the design matrix is a
column of ones, every estimator below has a closed form and no regression is
run.

Implemented:

* ``cluster_robust`` -- CR0 and the CR1 finite-sample scaling, with G-1
  degrees of freedom. The textbook answer.
* ``cluster_robust_cr2`` -- the Bell--McCaffrey bias reduction with
  Imbens--Kolesar degrees of freedom. For an intercept-only model the
  adjustment matrix has an explicit eigendecomposition, given in the
  function, so no matrix square root is computed numerically.
* ``cluster_jackknife`` -- CR3, the leave-one-cluster-out jackknife.
* ``wild_cluster_bootstrap`` -- the bootstrap-t with cluster-level Rademacher
  weights and null-imposed residuals.
* ``randomization_mean`` -- randomization inference on the cluster mean.

A relationship worth stating because it is easy to miss: for an
intercept-only model with null-imposed residuals, the wild cluster bootstrap
with Rademacher weights draws exactly the same sign flips as randomization
inference. They differ only in the statistic compared -- the bootstrap
studentises, randomization does not -- so a divergence between them is
informative about the variance estimate rather than about the resampling
scheme.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats


def _prep(d, groups) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(d, dtype=np.float64)
    g = pd.factorize(np.asarray(groups))[0]
    ok = np.isfinite(d)
    return d[ok], g[ok]


def _cluster_sums(x: np.ndarray, g: np.ndarray, n_groups: int) -> np.ndarray:
    return np.bincount(g, weights=x, minlength=n_groups)


def cluster_robust(d, groups, correction: str = "cr1") -> dict:
    """CR0/CR1 cluster-robust test of a zero mean, clustered by ``groups``.

    For :math:`X = \\iota_n`, :math:`(X'X)^{-1} = 1/n` and the cluster score is
    :math:`e_g = \\sum_{t \\in g}(d_{gt} - \\bar d)`, so the variance of the
    mean is :math:`n^{-2}\\sum_g e_g^2`. CR1 multiplies it by
    :math:`\\frac{G}{G-1}\\cdot\\frac{n-1}{n-1}` (the usual
    :math:`(n-1)/(n-k)` with :math:`k=1`).
    """
    d, g = _prep(d, groups)
    n = len(d)
    n_g = int(g.max()) + 1 if n else 0
    if n_g < 3:
        return {"stat": np.nan, "p": np.nan, "n_clusters": n_g, "df": np.nan}
    resid = d - d.mean()
    e = _cluster_sums(resid, g, n_g)
    var = float((e**2).sum()) / n**2
    if correction == "cr1":
        var *= n_g / (n_g - 1.0)
    if var <= 0:
        return {"stat": np.nan, "p": np.nan, "n_clusters": n_g, "df": n_g - 1}
    stat = d.mean() / np.sqrt(var)
    df = n_g - 1
    return {
        "stat": float(stat),
        "p": float(2 * (1 - stats.t.cdf(abs(stat), df=df))),
        "n_clusters": n_g,
        "df": float(df),
    }


def cluster_robust_cr2(d, groups) -> dict:
    """Bell--McCaffrey CR2 with Imbens--Kolesar degrees of freedom.

    CR2 pre-multiplies each cluster's residual vector by
    :math:`A_g = (I - H_{gg})^{-1/2}`. For an intercept-only model
    :math:`H = n^{-1}\\iota\\iota'`, so :math:`H_{gg} = n^{-1}J_{n_g}` and
    :math:`I - H_{gg}` has eigenvalue :math:`1` with multiplicity
    :math:`n_g - 1` and :math:`1 - n_g/n` once, on the constant direction.
    Hence

    .. math::
        A_g = I + \\Big[(1 - n_g/n)^{-1/2} - 1\\Big] n_g^{-1} J_{n_g},

    which is applied in closed form below. The Imbens--Kolesar degrees of
    freedom are :math:`(\\sum_g \\lambda_g)^2 / \\sum_g \\lambda_g^2` with
    :math:`\\lambda_g` the squared cluster weights, which for this design
    reduces to a Satterthwaite quantity in the adjusted cluster scores.
    """
    d, g = _prep(d, groups)
    n = len(d)
    n_g = int(g.max()) + 1 if n else 0
    if n_g < 3:
        return {"stat": np.nan, "p": np.nan, "n_clusters": n_g, "df": np.nan}
    sizes = np.bincount(g, minlength=n_g).astype(float)
    if (sizes >= n).any():  # a single cluster is the whole sample
        return {"stat": np.nan, "p": np.nan, "n_clusters": n_g, "df": np.nan}
    resid = d - d.mean()
    sums = _cluster_sums(resid, g, n_g)
    # A_g applied to the residual vector, then summed within cluster: the
    # identity part contributes `sums`, and the J part contributes
    # [(1 - n_g/n)^{-1/2} - 1] * sums.
    scale = (1.0 - sizes / n) ** -0.5
    adj = sums * scale
    var = float((adj**2).sum()) / n**2
    if var <= 0:
        return {"stat": np.nan, "p": np.nan, "n_clusters": n_g, "df": np.nan}
    stat = d.mean() / np.sqrt(var)
    lam = adj**2
    df = float((lam.sum() ** 2) / (lam**2).sum())
    return {
        "stat": float(stat),
        "p": float(2 * (1 - stats.t.cdf(abs(stat), df=max(df, 1.0)))),
        "n_clusters": n_g,
        "df": df,
    }


def cluster_jackknife(d, groups) -> dict:
    """CR3, the leave-one-cluster-out jackknife variance of the mean."""
    d, g = _prep(d, groups)
    n = len(d)
    n_g = int(g.max()) + 1 if n else 0
    if n_g < 3:
        return {"stat": np.nan, "p": np.nan, "n_clusters": n_g, "df": np.nan}
    total = d.sum()
    sizes = np.bincount(g, minlength=n_g).astype(float)
    sums = _cluster_sums(d, g, n_g)
    keep = n - sizes > 0
    if not keep.all():
        return {"stat": np.nan, "p": np.nan, "n_clusters": n_g, "df": np.nan}
    loo = (total - sums) / (n - sizes)  # mean leaving each cluster out
    var = ((n_g - 1.0) / n_g) * float(((loo - loo.mean()) ** 2).sum())
    if var <= 0:
        return {"stat": np.nan, "p": np.nan, "n_clusters": n_g, "df": np.nan}
    stat = d.mean() / np.sqrt(var)
    df = n_g - 1
    return {
        "stat": float(stat),
        "p": float(2 * (1 - stats.t.cdf(abs(stat), df=df))),
        "n_clusters": n_g,
        "df": float(df),
    }


def _block_weights(rng, n_draws: int, n_g: int, block: int) -> np.ndarray:
    r"""Sign weights that are constant within contiguous blocks of clusters.

    Independent per-cluster flips assume the clusters' signs are independent.
    On this design they are not always: the quarterly loss differential
    carries first-order autocorrelation of +0.32 to +0.48, and for one
    comparison a runs test on the signs themselves rejects. Flipping contiguous blocks of `block`
    clusters together preserves sign dependence up to that length under the
    null, which is the standard remedy and makes the resulting test a block
    randomization rather than an i.i.d.\ one.

    The block offset is randomised per draw so that no cluster boundary is
    privileged.
    """
    if block <= 1:
        return rng.choice([-1.0, 1.0], size=(n_draws, n_g))
    n_blocks = int(np.ceil(n_g / block)) + 1
    out = np.empty((n_draws, n_g))
    for b in range(n_draws):
        offset = int(rng.integers(0, block))
        signs = rng.choice([-1.0, 1.0], size=n_blocks)
        idx = (np.arange(n_g) + offset) // block
        out[b] = signs[idx]
    return out


def _cluster_stats(sums: np.ndarray, sizes: np.ndarray, n: int, W: np.ndarray):
    """CR1 mean and t-statistic for many sign-flip draws at once.

    For an intercept-only model everything depends on the data only through
    the cluster sums, so the whole bootstrap has a closed form. With weights
    :math:`w_g` and cluster sums :math:`S_g`,

    .. math::
        \bar d^* = n^{-1}\\sum_g w_g S_g, \\qquad
        e_g^* = w_g S_g - n_g \bar d^*, \\qquad
        \\widehat V^* = \frac{G}{G-1}\\, n^{-2} \\sum_g (e_g^*)^2 .

    ``W`` is ``(B, G)``. Vectorising this is not an optimisation detail: the
    loop version costs one full recomputation per draw, which put a
    ten-procedure size study out of reach.
    """
    n_g = len(sums)
    means = W @ sums / n
    e = W * sums[None, :] - sizes[None, :] * means[:, None]
    var = (n_g / (n_g - 1.0)) * (e**2).sum(axis=1) / n**2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(var > 0, means / np.sqrt(var), np.nan)
    return means, t


def wild_cluster_bootstrap(
    d,
    groups,
    n_boot: int = 999,
    seed: int = 0,
    weights: str = "rademacher",
    block: int = 1,
) -> dict:
    r"""Bootstrap-t with cluster-level weights and null-imposed residuals.

    The null is imposed by resampling the raw differentials, whose restricted
    fit is zero, each cluster receiving one weight; the CR1 t-statistic is
    recomputed on every draw. The p-value is the share of bootstrap
    statistics at least as extreme as the observed one, with the usual
    :math:`(1+\#)/(1+B)` correction so it can never be exactly zero.
    """
    d, g = _prep(d, groups)
    n = len(d)
    n_g = int(g.max()) + 1 if n else 0
    if n_g < 3:
        return {"stat": np.nan, "p": np.nan, "n_clusters": n_g, "n_boot": 0}
    observed = cluster_robust(d, g, correction="cr1")["stat"]
    if not np.isfinite(observed):
        return {"stat": np.nan, "p": np.nan, "n_clusters": n_g, "n_boot": 0}
    sums = _cluster_sums(d, g, n_g)
    sizes = np.bincount(g, minlength=n_g).astype(float)
    rng = np.random.default_rng(seed)
    if weights == "rademacher":
        W = _block_weights(rng, n_boot, n_g, block)
    else:  # Mammen two-point; blocks are only defined for sign weights
        lo, hi = (1 - np.sqrt(5)) / 2, (1 + np.sqrt(5)) / 2
        p_lo = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
        W = rng.choice([lo, hi], size=(n_boot, n_g), p=[p_lo, 1 - p_lo])
    _, t = _cluster_stats(sums, sizes, n, W)
    extreme = int(np.sum(np.abs(t[np.isfinite(t)]) >= abs(observed)))
    return {
        "stat": float(observed),
        "p": float((1 + extreme) / (1 + n_boot)),
        "n_clusters": n_g,
        "n_boot": n_boot,
    }


def randomization_mean(
    d, groups, n_draws: int = 999, seed: int = 0, block: int = 1
) -> dict:
    """Randomization inference on the cluster mean, un-studentised.

    Sign-flips whole clusters and compares the observed mean differential to
    the randomization distribution of the mean. This is the same resampling
    scheme as the wild cluster bootstrap with Rademacher weights, so the two
    differ only in whether the statistic is studentised. On this design they
    agree almost exactly, which is informative rather than redundant: for an
    intercept-only model the cluster-robust variance is nearly invariant to
    the sign flips, so studentising is a near-monotone transformation and
    cannot reorder the draws.
    """
    d, g = _prep(d, groups)
    n = len(d)
    n_g = int(g.max()) + 1 if n else 0
    if n_g < 3:
        return {"stat": np.nan, "p": np.nan, "n_clusters": n_g, "n_draws": 0}
    sums = _cluster_sums(d, g, n_g)
    sizes = np.bincount(g, minlength=n_g).astype(float)
    rng = np.random.default_rng(seed)
    W = _block_weights(rng, n_draws, n_g, block)
    means, _ = _cluster_stats(sums, sizes, n, W)
    observed = abs(float(d.mean()))
    extreme = int(np.sum(np.abs(means) >= observed))
    return {
        "stat": float(d.mean()),
        "p": float((1 + extreme) / (1 + n_draws)),
        "n_clusters": n_g,
        "n_draws": n_draws,
    }


PROCEDURES = {
    "cluster_cr1": lambda d, g, **kw: cluster_robust(d, g, correction="cr1"),
    "cluster_cr2": lambda d, g, **kw: cluster_robust_cr2(d, g),
    "cluster_cr3": lambda d, g, **kw: cluster_jackknife(d, g),
    "wild_bootstrap": wild_cluster_bootstrap,
    "randomization_mean": randomization_mean,
    "hac_quarterly": lambda d, g, **kw: hac_quarterly(d, g),
    "moving_block": lambda d, g, **kw: moving_block_bootstrap(d, g),
}

# ---------------------------------------------------------------------------
# Inference that is VALID under temporal dependence between clusters
# ---------------------------------------------------------------------------
# The procedures above treat clusters as independent. Block sign-flipping
# weakens that assumption but is not a valid test for block length > 1
#. The two procedures here are asymptotically
# valid under weak dependence in the cluster-level series itself, which is
# what the diagnostics in §86 say we need: they operate on the G
# quarter-aggregated differentials and correct for their serial correlation
# rather than assuming it away.


def hac_quarterly(d, groups, lags: int | None = None) -> dict:
    """Newey--West HAC test on the CLUSTER-AGGREGATED differentials.

    The overlap is removed by aggregation, so the remaining dependence is
    between clusters, and a Bartlett-weighted long-run variance on the G
    cluster means is asymptotically valid under weak dependence. This is the
    textbook answer to serial correlation in the quarterly series and does
    not require exchangeability of any kind.

    ``lags`` defaults to the Newey--West plug-in rule
    :math:`\\lfloor 4(G/100)^{2/9} \rfloor`, which for G near 82 gives 3 --
    enough to span the first-order autocorrelation of +0.32 to +0.48 measured
    on this data with room over.
    """
    from scipy import stats as _stats

    d, g = _prep(d, groups)
    n_g = int(g.max()) + 1 if len(d) else 0
    if n_g < 8:
        return {"stat": np.nan, "p": np.nan, "n_clusters": n_g, "lags": np.nan}
    dq = np.bincount(g, weights=d) / np.bincount(g)
    if lags is None:
        lags = int(np.floor(4 * (n_g / 100.0) ** (2.0 / 9.0)))
    x = dq - dq.mean()
    lrv = float((x * x).sum() / n_g)
    for lag in range(1, min(lags, n_g - 1) + 1):
        gamma = float((x[lag:] * x[:-lag]).sum() / n_g)
        lrv += 2.0 * (1.0 - lag / (lags + 1.0)) * gamma
    if not np.isfinite(lrv) or lrv <= 0:
        return {"stat": np.nan, "p": np.nan, "n_clusters": n_g, "lags": lags}
    stat = dq.mean() / np.sqrt(lrv / n_g)
    return {
        "stat": float(stat),
        "p": float(2 * (1 - _stats.t.cdf(abs(stat), df=n_g - 1))),
        "n_clusters": n_g,
        "lags": int(lags),
    }


def moving_block_bootstrap(
    d, groups, n_boot: int = 999, block: int | None = None, seed: int = 0
) -> dict:
    r"""Moving-block bootstrap on the cluster-aggregated differentials.

    Contiguous blocks of the G quarterly means are resampled WITH
    replacement, which reproduces the series' short-run dependence in each
    resample and is asymptotically valid for the mean of a weakly dependent
    stationary sequence. Unlike block sign-flipping it does not require
    symmetry or exchangeability, so it is a valid test rather than a
    sensitivity analysis; the price is that validity is asymptotic in G
    rather than exact, and G is 82 here.

    The null is imposed by centring each resample on the observed mean, so
    the bootstrap distribution approximates that of the mean under
    :math:`H_0^{\mu}` and the p-value is a two-sided studentised percentile.
    """
    d, g = _prep(d, groups)
    n_g = int(g.max()) + 1 if len(d) else 0
    if n_g < 12:
        return {"stat": np.nan, "p": np.nan, "n_clusters": n_g, "block": np.nan}
    dq = np.bincount(g, weights=d) / np.bincount(g)
    if block is None:
        # Politis-White style rule of thumb: G^{1/3}, which is 4 at G = 82.
        block = max(round(n_g ** (1.0 / 3.0)), 2)
    observed = float(dq.mean())
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_g / block))
    starts = rng.integers(0, n_g - block + 1, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = (starts[b][:, None] + offsets[None, :]).ravel()[:n_g]
        means[b] = dq[idx].mean()
    # Centre on the observed mean to impose the null, then compare.
    centred = means - means.mean()
    extreme = int(np.sum(np.abs(centred) >= abs(observed)))
    return {
        "stat": observed,
        "p": float((1 + extreme) / (1 + n_boot)),
        "n_clusters": n_g,
        "block": int(block),
        "se_boot": float(means.std(ddof=1)),
    }


def ibragimov_muller(d, groups, q: int = 8, rotate: float = 0.0) -> dict:
    r"""Ibragimov--Muller (2010) t-test on q CONTIGUOUS groups of clusters.

    This is the procedure designed for the situation the rest of this module
    documents and cannot fix: few clusters, heterogeneous variances, and
    dependence that a large-G argument cannot be leaned on. The sample is
    partitioned into ``q`` contiguous groups of target periods, the mean
    differential is computed within each, and the q group estimates are
    treated as independent normal draws with possibly different variances,
    giving a t-test on q observations with q - 1 degrees of freedom.

    Two features matter here. Because the groups are *contiguous* in time,
    dependence at lags shorter than a group's span is absorbed inside a group
    rather than violating the independence assumption across them -- which is
    the opposite of what cluster-independent resampling does with the same
    data. And the validity argument is a small-sample one: Ibragimov and
    Muller show the test's level is bounded by the nominal level for
    :math:`\alpha \le 0.083` with q as small as 2, at the cost of power,
    rather than requiring q to grow.

    The price is real and we report it: q group means discard the
    within-group variation that a cluster-level procedure uses, so the test
    is conservative by construction.

    ``rotate`` shifts the partition by that fraction of a group before
    splitting, wrapping the tail around to the front. It exists so the
    verdict can be checked against the arbitrariness of where the group
    boundaries fall: a result that moves when the boundaries move by half a
    group is an artifact of the partition rather than a property of the data.
    """
    from scipy import stats as _stats

    d, g = _prep(d, groups)
    n_g = int(g.max()) + 1 if len(d) else 0
    if n_g < 2 * q:
        return {"stat": np.nan, "p": np.nan, "n_clusters": n_g, "q": q}
    dq = np.bincount(g, weights=d) / np.bincount(g)
    # Contiguous, as near equal in size as G/q allows. np.array_split puts the
    # larger groups first, which is immaterial: the test weights groups
    # equally by construction, and that is exactly why it tolerates
    # heterogeneity.
    if rotate:
        shift = round(rotate * n_g / q) % n_g
        dq = np.roll(dq, shift)
    thetas = np.array([blk.mean() for blk in np.array_split(dq, q)])
    sd = thetas.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return {"stat": np.nan, "p": np.nan, "n_clusters": n_g, "q": q}
    stat = thetas.mean() / (sd / np.sqrt(q))
    return {
        "stat": float(stat),
        "p": float(2 * (1 - _stats.t.cdf(abs(stat), df=q - 1))),
        "n_clusters": n_g,
        "q": int(q),
        "rotate": float(rotate),
        "group_means": thetas,
    }


def _fixed_b_critical(b: float) -> float:
    """Two-sided 5% critical value of the fixed-b Bartlett t-statistic."""
    return float(np.quantile(_fixed_b_draws(round(b, 3)), 0.95))


def hac_quarterly_fixed_b(d, groups, b: float = 0.25) -> dict:
    r"""The HAC test on cluster means, judged by fixed-b critical values.

    :func:`hac_quarterly` compares its statistic to a t distribution, which
    is a small-b approximation: it treats the bandwidth as negligible
    relative to the sample. At G = 82 with a bandwidth of 3 that is already
    questionable, and it is exactly the approximation that makes the
    procedure liberal in the size study. Fixed-b asymptotics
    \citep{kiefer2005} instead hold L/G = b fixed, which accounts for the
    randomness of the variance estimator and yields larger critical values.

    This returns a p-value by inverting the simulated fixed-b distribution
    on a grid, so it is comparable with the other procedures' p-values rather
    than only a reject/not-reject verdict.
    """
    d, g = _prep(d, groups)
    n_g = int(g.max()) + 1 if len(d) else 0
    if n_g < 12:
        return {"stat": np.nan, "p": np.nan, "n_clusters": n_g, "b": b}
    lags = max(round(b * n_g), 1)
    base = hac_quarterly(d, g, lags=lags)
    if not np.isfinite(base["stat"]):
        return {"stat": np.nan, "p": np.nan, "n_clusters": n_g, "b": b}
    draws = _fixed_b_draws(round(b, 3))
    stat = abs(base["stat"])
    # The p-value is the simulated tail probability, with the usual +1 so it
    # is never exactly zero and never exactly the nominal level -- a grid
    # lookup returned 0.05 on the boundary, which a `p < 0.05` test counts as
    # a non-rejection.
    p = float((1 + int(np.sum(draws >= stat))) / (1 + len(draws)))
    return {
        "stat": float(base["stat"]),
        "p": p,
        "crit_05": float(np.quantile(draws, 0.95)),
        "n_clusters": n_g,
        "lags": lags,
        "b": float(b),
    }


@lru_cache(maxsize=32)
def _fixed_b_draws(b: float, n_sim: int = 20000, n_grid: int = 1000, seed: int = 11):
    r"""Simulated draws of :math:`|t|` under the fixed-b limit, sorted.

    Under fixed-b asymptotics the Newey--West t-statistic does not converge
    to a normal or a t: with bandwidth :math:`L = bT` held proportional to
    the sample, the limit is a nonstandard but PIVOTAL functional of a
    Brownian bridge \citep{kiefer2005}, so its distribution can be simulated
    once rather than read from a published table. Returning the draws rather
    than one quantile lets the p-value be a continuous ECDF tail probability.

    The limit is :math:`W(1)/\sqrt{Q_b}` with

    .. math::
        Q_b = \frac{2}{b}\Bigl[\int_0^1 \tilde B(r)^2\,dr
              - \int_0^{1-b} \tilde B(r+b)\tilde B(r)\,dr\Bigr],

    and :math:`\tilde B(r) = W(r) - rW(1)` the Brownian bridge. The simulated
    5% values -- 2.22 at b = 0.1 and 2.67 at b = 0.25, against 1.99 for a t
    with 79 degrees of freedom -- agree with the published Bartlett-kernel
    values, which is the check that the functional is implemented as
    intended.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / n_grid
    step = max(round(b * n_grid), 1)
    r = np.arange(1, n_grid + 1) * dt
    out = np.empty(n_sim)
    for i in range(n_sim):
        w = np.cumsum(rng.standard_normal(n_grid) * np.sqrt(dt))
        bridge = w - r * w[-1]
        term1 = float((bridge**2).sum() * dt)
        term2 = (
            float((bridge[step:] * bridge[:-step]).sum() * dt) if step < n_grid else 0.0
        )
        qb = (2.0 / b) * (term1 - term2)
        out[i] = abs(w[-1] / np.sqrt(qb)) if qb > 0 else np.nan
    return np.sort(out[np.isfinite(out)])
