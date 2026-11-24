"""Diebold-Mariano (1995) predictive accuracy test with HAC variance.

Implements docs/02_methodology.md §4. Used pairwise against the DFM
benchmark (paper appendix C).

Two corrections matter for this design and are applied by default:

* NEWEY-WEST long-run variance. Overlapping nowcast dates for the same
  target quarter make the loss differential strongly autocorrelated -- there
  are ~17 weekly nowcasts per quarter, all predicting the same number -- so
  the naive DM variance is far too small and would manufacture significance.
  ``hac_lags`` comes from ``evaluation.dm_hac_lags`` in the config.
* HARVEY-LEYBOURNE-NEWBOLD (1997) small-sample correction. The sample here
  is ~80 quarters, not thousands, and the uncorrected DM statistic
  over-rejects badly at that size. The corrected statistic is compared
  against a t distribution with n-1 degrees of freedom rather than the
  normal.

WARNING -- ``hac_lags`` is not a free tuning knob.
This design has ~17 weekly nowcast dates per target quarter, all predicting
the SAME number, and ``dm_test`` counts each of them as an observation. With
``hac_lags=4`` -- the value this study used until §66 -- six of nine headline
comparisons came out significant at 5%; at ``hac_lags=17``, the length of the
actual overlap, NONE of them did, and the same applies to averaging the loss
differential to one observation per target quarter. The n in a weekly DM
statistic is 1,388; the number of independent target draws is 82.

``dm_test_by_quarter`` is therefore the REPORTED inference for this design:
it collapses the overlap by construction and adds a sign test and a signed-
rank test, both of which matter because the loss differentials are extremely
fat-tailed (one COVID quarter can carry more squared error than a decade of
normal ones). Use ``dm_test`` with an explicit ``hac_lags`` only for the
robustness curve that documents the sensitivity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _loss(e: np.ndarray, loss: str) -> np.ndarray:
    if loss == "squared":
        return e**2
    if loss == "absolute":
        return np.abs(e)
    raise ValueError(f"unsupported loss: {loss!r}")


def _newey_west_variance(d: np.ndarray, hac_lags: int) -> float:
    """Long-run variance of the mean of ``d`` with Bartlett weights.

    Uses explicit ``np.sum`` products rather than ``d @ d``: numpy 2.0.2 on
    Apple Silicon (Accelerate BLAS) raises a SPURIOUS "divide by zero
    encountered in matmul" on any moderately sized float64 matmul --
    reproduced with 1,403 random normals, where ``np.dot`` and
    ``np.sum(d*d)`` return the identical correct value without warning. A
    stray warning in the inference path invites doubt about the numbers, so
    the arithmetic is done in a form that does not emit one.
    """
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    d = d - d.mean()
    gamma0 = float(np.sum(d * d) / n)
    total = gamma0
    for lag in range(1, hac_lags + 1):
        if lag >= n:
            break
        gamma = float(np.sum(d[lag:] * d[:-lag]) / n)
        weight = 1.0 - lag / (hac_lags + 1.0)  # Bartlett
        total += 2.0 * weight * gamma
    return total


def dm_test(
    e1: pd.Series,
    e2: pd.Series,
    h: int = 1,
    loss: str = "squared",
    hac_lags: int | None = None,
    hln_correction: bool = True,
):
    """Return (DM statistic, p-value) for equal predictive accuracy.

    H0: equal accuracy. A NEGATIVE statistic means ``e1``'s model is more
    accurate (its loss is lower). Two-sided p-value.

    ``e1``/``e2`` must be aligned errors for the same forecast pairs; the
    caller is responsible for having restricted them to a common sample.
    """
    a = np.asarray(e1, dtype=np.float64)
    b = np.asarray(e2, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"error series must align: {a.shape} vs {b.shape}")
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 8:
        return float("nan"), float("nan")

    d = (_loss(a, loss) - _loss(b, loss)).astype(np.float64)
    if np.allclose(d, 0):
        return 0.0, 1.0

    if hac_lags is None:
        # The conventional h-step default is h-1, which for a nowcast is ZERO:
        # no autocorrelation correction at all, the maximally anti-conservative
        # case on an overlapping design ( measures the
        # size of the merely-too-short case at up to 51%). Defaulting to it
        # silently is a landmine, so the bandwidth must be chosen explicitly by
        # the caller, from the evaluation grid via `hac_lags_for_overlap`.
        raise ValueError(
            "hac_lags must be given explicitly: the h-1 convention yields 0 "
            "for a nowcast, which applies no autocorrelation correction to an "
            "overlapping sample. Derive it from the design with "
            "hac_lags_for_overlap(dates_per_target), or use "
            "dm_test_by_quarter, which removes the overlap by construction "
            "and is the reported inference for this design."
        )

    lrv = _newey_west_variance(d, hac_lags)
    if not np.isfinite(lrv) or lrv <= 0:
        return float("nan"), float("nan")

    stat = d.mean() / np.sqrt(lrv / n)

    if hln_correction:
        # Harvey, Leybourne & Newbold (1997): shrink the statistic and use a
        # t reference distribution. Without it, ~80 quarters over-reject.
        factor = (n + 1 - 2 * h + h * (h - 1) / n) / n
        factor = max(factor, 1e-8)
        stat = stat * np.sqrt(factor)
        pvalue = 2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1))
    else:
        pvalue = 2.0 * (1.0 - stats.norm.cdf(abs(stat)))

    return float(stat), float(pvalue)


def hac_lags_for_overlap(dates_per_target: int) -> int:
    """Newey-West bandwidth implied by the overlap, not by convention.

    With ``dates_per_target`` nowcast dates aimed at the same target, the
    loss differential is autocorrelated out to that many lags, so the
    bandwidth must be at least that long. Returning it from a function makes
    the assumption visible instead of leaving a 4 in a config file.
    """
    return max(int(dates_per_target), 1)


def dm_test_by_quarter(
    e1: pd.Series,
    e2: pd.Series,
    quarters=None,
    loss: str = "squared",
):
    """Predictive-accuracy comparison with the intra-quarter overlap removed.

    The loss differential is averaged within each target quarter, so each
    target contributes exactly one observation and the ~17-fold overlap of
    the weekly sample cannot inflate the statistic. Three tests are returned
    because they fail differently:

    * ``p_t``       -- t-test on the quarterly mean differential. Sensitive
                       to the fat tails: a single crisis quarter dominates.
    * ``p_sign``    -- exact binomial sign test. Answers "in how many
                       quarters is the model ahead?", ignoring magnitude, so
                       it detects a model that usually wins by a little.
    * ``p_wilcoxon``-- signed-rank test. Uses magnitude ordinally; the
                       compromise between the two.

    A NEGATIVE ``t_stat`` means ``e1`` is more accurate, matching
    ``dm_test``'s sign convention. ``quarters`` defaults to the
    ``target_quarter`` level of the inputs' shared index.
    """
    frame = pd.concat({"e1": e1, "e2": e2}, axis=1).dropna()
    if quarters is None:
        if "target_quarter" not in (frame.index.names or []):
            raise ValueError("pass `quarters` or index the errors by target_quarter")
        key = frame.index.get_level_values("target_quarter")
    else:
        key = pd.Series(quarters).reindex(frame.index).to_numpy()

    d = (
        pd.Series(
            _loss(frame["e1"].to_numpy(float), loss)
            - _loss(frame["e2"].to_numpy(float), loss),
            index=frame.index,
        )
        .groupby(key)
        .mean()
    )
    n = len(d)
    out = {
        "n_pairs": len(frame),
        "n_quarters": n,
        "rmse_1": float(np.sqrt((frame["e1"] ** 2).mean())),
        "rmse_2": float(np.sqrt((frame["e2"] ** 2).mean())),
        "mean_loss_diff": float(d.mean()),
        "n_quarters_model_ahead": int((d < 0).sum()),
    }
    if n < 8 or d.std(ddof=1) == 0:
        out.update(
            t_stat=float("nan"),
            p_t=float("nan"),
            p_sign=float("nan"),
            p_wilcoxon=float("nan"),
        )
        return out

    t_stat = float(d.mean() / (d.std(ddof=1) / np.sqrt(n)))
    out["t_stat"] = t_stat
    out["p_t"] = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1)))
    ahead = int((d < 0).sum())
    out["p_sign"] = float(stats.binomtest(ahead, n, 0.5).pvalue)
    try:
        out["p_wilcoxon"] = float(stats.wilcoxon(d.to_numpy(float))[1])
    except ValueError:  # all differentials identical
        out["p_wilcoxon"] = float("nan")
    return out


def hac_sensitivity(
    e1: pd.Series,
    e2: pd.Series,
    lags=(4, 8, 17, 26, 52),
    loss: str = "squared",
) -> pd.DataFrame:
    """DM statistic and p-value across HAC bandwidths.

    Reported so a reader can see how much of a result is the bandwidth
    choice. For this design the honest answer is: most of it (§66).
    """
    rows = []
    for lag in lags:
        stat, p = dm_test(e1, e2, loss=loss, hac_lags=lag)
        rows.append({"hac_lags": lag, "dm_stat": stat, "p_value": p})
    return pd.DataFrame(rows)


def dm_table(
    prepared: pd.DataFrame,
    benchmark: str = "dfm",
    loss: str = "squared",
    hac_lags: int | None = None,
    by_block: bool = True,
):
    """Pairwise DM against ``benchmark``, per model and cycle block (RQ2).

    ``prepared`` is the output of ``metrics.common_sample(metrics.prepare(...))``
    so that every model is compared on identical forecast pairs.

    Reports BOTH inferences: the weekly HAC statistic under
    ``*_weekly_hac`` names, kept as the audit trail, and the quarterly t,
    sign and signed-rank tests, which are what the ``significant_5pct_*``
    columns reflect.
    """
    keys = ["nowcast_date", "target_quarter"]
    wide = prepared.pivot_table(index=keys, columns="model", values="error")
    if benchmark not in wide.columns:
        raise ValueError(f"benchmark {benchmark!r} not among {list(wide.columns)}")

    if hac_lags is None:
        # Derived from the evaluation grid rather than fixed: the bandwidth
        # has to match the overlap this table's own sample carries, and a
        # hard-coded 4 is the error the accompanying paper is about.
        n_targets = wide.index.get_level_values("target_quarter").nunique()
        hac_lags = hac_lags_for_overlap(round(len(wide) / max(n_targets, 1)))

    groups = {"all": prepared}
    if by_block and "block" in prepared.columns:
        for block, g in prepared.groupby("block"):
            groups[block] = g

    rows = []
    for block, g in groups.items():
        w = g.pivot_table(index=keys, columns="model", values="error").dropna()
        if benchmark not in w.columns or len(w) < 8:
            continue
        for model in w.columns:
            if model == benchmark:
                continue
            stat, p = dm_test(w[model], w[benchmark], loss=loss, hac_lags=hac_lags)
            # The weekly HAC p-value is anti-conservative on this design
            #, so it is carried under an explicit
            # name and the quarterly tests decide the verdict columns.
            q = dm_test_by_quarter(w[model], w[benchmark], loss=loss)
            rows.append(
                {
                    "block": block,
                    "model": model,
                    "benchmark": benchmark,
                    "n": len(w),
                    "n_quarters": q["n_quarters"],
                    "dm_stat_weekly_hac": stat,
                    "p_weekly_hac": p,
                    "t_stat_quarterly": q["t_stat"],
                    "p_t_quarterly": q["p_t"],
                    "quarters_model_ahead": q["n_quarters_model_ahead"],
                    "p_sign": q["p_sign"],
                    "p_wilcoxon": q["p_wilcoxon"],
                    "better_than_benchmark": (
                        bool(q["t_stat"] < 0) if np.isfinite(q["t_stat"]) else None
                    ),
                    # Named per test: a single `significant_5pct` column would
                    # hide that the t and sign tests disagree here -- the
                    # bridge is significant by sign and not by t (§66).
                    "significant_5pct_t": (
                        bool(q["p_t"] < 0.05) if np.isfinite(q["p_t"]) else None
                    ),
                    "significant_5pct_sign": (
                        bool(q["p_sign"] < 0.05) if np.isfinite(q["p_sign"]) else None
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["block", "model"]).reset_index(drop=True)
