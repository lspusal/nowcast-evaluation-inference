"""Model Confidence Set (Hansen, Lunde & Nason 2011).

Implements docs/02_methodology.md §4 (paper Table T2 / appendix C).

The MCS answers the question the study actually needs: not "is model A
better than the DFM" (that is the pairwise DM test) but "which models can we
NOT distinguish from the best at the chosen confidence level". With a ~80
quarter sample and a dozen models, that distinction is the difference
between an honest claim and an overclaim.

Procedure implemented: the T_max variant with a MOVING BLOCK bootstrap over
the loss differentials, eliminating the worst model at each step until the
equal-predictive-ability null is no longer rejected. Block bootstrap (rather
than iid) because overlapping weekly nowcasts for the same target quarter
make the losses strongly serially dependent -- the same reason the DM test
here needs HAC. See.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _moving_block_indices(
    n: int, block: int, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """(n_boot x n) resampled time indices from overlapping blocks."""
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block + 1, 1), size=(n_boot, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1)
    return np.minimum(idx[:, :n], n - 1)


def mcs(
    losses: pd.DataFrame,
    alpha: float = 0.10,
    n_boot: int = 5000,
    block: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    """Model Confidence Set via the T_max statistic and a block bootstrap.

    ``losses`` is (T x models), aligned in time with no missing values --
    every model must be evaluated on identical forecast pairs, otherwise the
    comparison is not of accuracy but of which dates each model attempted.

    Returns one row per model with its MCS p-value and whether it survives at
    ``alpha``. The p-value is the level at which the model would be
    eliminated, so larger means "harder to reject as being the best".
    """
    losses = losses.dropna()
    if losses.shape[1] < 2 or len(losses) < 8:
        raise ValueError(f"need >=2 models and >=8 observations, got {losses.shape}")

    rng = np.random.default_rng(seed)
    n = len(losses)
    boot_idx = _moving_block_indices(n, block, n_boot, rng)

    alive = list(losses.columns)
    pvalues: dict[str, float] = {}
    running_max = 0.0

    while len(alive) > 1:
        L = losses[alive].to_numpy()

        means = L.mean(axis=0)

        # Bootstrap distribution of each model's mean loss.
        boot_means = L[boot_idx, :].mean(axis=1)  # (n_boot x m)
        centred = boot_means - means

        # Pairwise differentials d_ij = mean_i - mean_j.
        d = means[:, None] - means[None, :]
        boot_d = centred[:, :, None] - centred[:, None, :]
        var_d = boot_d.var(axis=0, ddof=1)
        np.fill_diagonal(var_d, np.inf)  # avoid 0/0 on the diagonal
        var_d = np.where(var_d <= 0, np.inf, var_d)

        t = d / np.sqrt(var_d)
        boot_t = boot_d / np.sqrt(var_d)[None, :, :]

        # T_max: the largest standardised excess loss over the set.
        t_max = np.nanmax(t, axis=1)
        boot_t_max = np.nanmax(np.nanmax(boot_t, axis=2), axis=1)
        stat = float(np.nanmax(t_max))
        pval = float(np.mean(boot_t_max >= stat))

        # MCS p-values are monotone by construction: a model eliminated later
        # cannot have a smaller p-value than one eliminated earlier.
        running_max = max(running_max, pval)

        if pval >= alpha:
            for name in alive:
                pvalues[name] = running_max
            break

        worst = alive[int(np.nanargmax(t_max))]
        pvalues[worst] = running_max
        alive.remove(worst)
        logger.debug("MCS: eliminated %s (p=%.4f)", worst, running_max)
    else:
        pvalues[alive[0]] = 1.0

    rows = [
        {
            "model": name,
            "mean_loss": float(losses[name].mean()),
            "mcs_pvalue": pvalues.get(name, np.nan),
            "in_mcs": bool(pvalues.get(name, 0.0) >= alpha),
        }
        for name in losses.columns
    ]
    return pd.DataFrame(rows).sort_values("mean_loss").reset_index(drop=True)


def mcs_table(
    prepared: pd.DataFrame,
    alpha: float = 0.10,
    loss: str = "squared",
    n_boot: int = 5000,
    block: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    """MCS over all models, from a prepared/common-sample prediction frame."""
    keys = ["nowcast_date", "target_quarter"]
    wide = prepared.pivot_table(index=keys, columns="model", values="error").dropna()
    if wide.shape[1] < 2:
        raise ValueError("need at least two models with a common sample")
    losses = wide**2 if loss == "squared" else wide.abs()
    return mcs(losses, alpha=alpha, n_boot=n_boot, block=block, seed=seed)
