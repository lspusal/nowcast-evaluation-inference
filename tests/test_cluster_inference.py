"""The cluster-level procedures, validated against external references.

The study's methodological claim is that a family of standard procedures
fails on this design. That claim is only as good as our implementation of
those procedures, so each is checked against a reference computation before
any conclusion rests on it.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.cluster_inference import (
    _block_weights,
    cluster_jackknife,
    cluster_robust,
    cluster_robust_cr2,
    randomization_mean,
    wild_cluster_bootstrap,
)


def _panel(n_groups=40, size=10, seed=0, effect=0.0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), size)
    d = rng.normal(size=n_groups * size) + rng.normal(size=n_groups)[g] + effect
    return d, g


def test_cr1_matches_statsmodels_cluster_inference():
    """CR1 must equal the cluster-robust t of a standard package."""
    sm = pytest.importorskip("statsmodels.api")
    d, g = _panel(seed=1)
    fitted = sm.OLS(d, np.ones((len(d), 1))).fit(
        cov_type="cluster", cov_kwds={"groups": g, "use_correction": True}
    )
    ours = cluster_robust(d, g, correction="cr1")["stat"]
    assert ours == pytest.approx(float(fitted.tvalues[0]), rel=1e-10)
    # We report G-1 degrees of freedom, the convention for cluster inference,
    # rather than statsmodels' n-k; the statistic is what must agree.
    assert ours != 0
    assert cluster_robust(d, g)["df"] == 39


def test_cr0_with_singleton_clusters_is_the_ordinary_statistic():
    d, g = _panel(n_groups=200, size=1, seed=2)
    ours = cluster_robust(d, g, correction="cr0")["stat"]
    reference = d.mean() / np.sqrt(((d - d.mean()) ** 2).sum() / len(d) ** 2)
    assert ours == pytest.approx(reference, rel=1e-12)


def test_cr3_matches_an_explicit_jackknife():
    d, g = _panel(n_groups=12, size=8, seed=3)
    n_g = 12
    loo = np.array([d[g != k].mean() for k in range(n_g)])
    var = ((n_g - 1) / n_g) * ((loo - loo.mean()) ** 2).sum()
    assert cluster_jackknife(d, g)["stat"] == pytest.approx(
        d.mean() / np.sqrt(var), rel=1e-12
    )


def test_cr1_equals_the_quarter_aggregated_t_on_a_balanced_grid():
    """The proposition in the study: with equal cluster sizes the CR1
    statistic IS the t-statistic on cluster means, so cluster-robust
    inference and quarter aggregation are one procedure (§85)."""
    for n_groups, size in ((20, 10), (82, 17), (50, 3)):
        d, g = _panel(n_groups, size, seed=n_groups)
        means = d.reshape(n_groups, size).mean(axis=1)
        t = means.mean() / (means.std(ddof=1) / np.sqrt(n_groups))
        assert cluster_robust(d, g)["stat"] == pytest.approx(
            t, rel=1e-11
        ), f"the equivalence fails at G={n_groups}, m={size}"


def test_the_equivalence_breaks_when_clusters_are_unbalanced():
    """Stated as an equality only for the balanced case, so the unbalanced
    case must actually differ -- otherwise the proposition is vacuous."""
    rng = np.random.default_rng(4)
    sizes = rng.integers(4, 30, size=30)
    g = np.repeat(np.arange(30), sizes)
    d = rng.normal(size=int(sizes.sum()))
    means = np.array([d[g == k].mean() for k in range(30)])
    t = means.mean() / (means.std(ddof=1) / np.sqrt(30))
    assert cluster_robust(d, g)["stat"] != pytest.approx(t, rel=1e-3)


def test_wild_bootstrap_vectorisation_matches_a_loop():
    """The closed form used for speed must reproduce the naive resampling."""
    d, g = _panel(n_groups=20, size=17, seed=5, effect=0.1)
    observed = cluster_robust(d, g, correction="cr1")["stat"]
    rng = np.random.default_rng(9)
    extreme, draws = 0, 500
    for _ in range(draws):
        w = rng.choice([-1.0, 1.0], size=20)
        star = cluster_robust(d * w[g], g, correction="cr1")["stat"]
        extreme += abs(star) >= abs(observed)
    by_loop = (1 + extreme) / (1 + draws)
    vectorised = wild_cluster_bootstrap(d, g, n_boot=draws, seed=9)["p"]
    assert vectorised == pytest.approx(by_loop, abs=1e-12)


def test_all_cluster_procedures_are_correctly_sized_under_a_gaussian_null():
    """Every procedure must hold its nominal level on the design's own
    cluster structure. This is what licenses using them at all, and it is
    what makes the real-data failure a statement about the data."""
    procedures = {
        "cr1": lambda d, g: cluster_robust(d, g)["p"],
        "cr2": lambda d, g: cluster_robust_cr2(d, g)["p"],
        "cr3": lambda d, g: cluster_jackknife(d, g)["p"],
        "wild": lambda d, g: wild_cluster_bootstrap(d, g, n_boot=199, seed=0)["p"],
        "randomization": lambda d, g: randomization_mean(d, g, n_draws=199, seed=0)[
            "p"
        ],
    }
    reps = 400
    g = np.repeat(np.arange(82), 17)
    draws = [
        np.random.default_rng(1000 + i).normal(size=82)[g]
        + np.random.default_rng(5000 + i).normal(size=82 * 17)
        for i in range(reps)
    ]
    for name, fn in procedures.items():
        rejections = sum(1 for d in draws if fn(d, g) < 0.05)
        size = rejections / reps
        assert 0.02 < size < 0.10, f"{name} rejects {size:.3f} at nominal 0.05"


def test_block_weights_are_constant_within_blocks():
    rng = np.random.default_rng(6)
    weights = _block_weights(rng, 200, 40, block=4)
    assert weights.shape == (200, 40)
    assert set(np.unique(weights)) <= {-1.0, 1.0}
    # Within a draw, runs of equal sign must average at least the block
    # length once the random offset is accounted for.
    runs = [1 + int((np.diff(row) != 0).sum()) for row in weights]
    assert np.mean(runs) < 40 / 2, "block weights are not blocked"
    independent = _block_weights(rng, 200, 40, block=1)
    runs_indep = [1 + int((np.diff(row) != 0).sum()) for row in independent]
    assert np.mean(runs_indep) > np.mean(runs)


def test_block_randomization_is_more_conservative_than_independent_flips():
    """Preserving sign dependence costs power, which is the point: a result
    that survives block randomization tolerates serial dependence."""
    rng = np.random.default_rng(7)
    n_groups = 82
    # A persistent effect: consecutive clusters share sign.
    base = np.repeat(rng.normal(0.4, 0.2, n_groups // 4 + 1), 4)[:n_groups]
    g = np.repeat(np.arange(n_groups), 10)
    d = base[g] + rng.normal(size=n_groups * 10)
    p_indep = wild_cluster_bootstrap(d, g, n_boot=999, seed=0, block=1)["p"]
    p_block = wild_cluster_bootstrap(d, g, n_boot=999, seed=0, block=4)["p"]
    assert p_block >= p_indep
