"""Evaluation metrics and result assembly (M6).

Implements docs/02_methodology.md §4. All comparisons against the SECOND
official release of GDP growth (R2) loaded from
data/processed/target_second_release.csv.

Evaluation rules enforced here, not left to the caller:

* Only GENUINE forecasts are scored. A pair whose answer was already public
  at the nowcast date (``target_already_published``, the design notes) is not a
  forecast, and scoring it would flatter every model on the backcast horizon.
* Abstentions (NaN ``yhat``) are counted and reported, never silently
  dropped: a model that abstains on the hard dates would otherwise look
  better than one that tries. ``coverage`` makes that visible.
* Models are compared on the INTERSECTION of dates they all cover, so a
  difference in RMSE cannot come from a difference in which dates were
  attempted.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# RQ2 cycle blocks, in days from the end of the target quarter.
CYCLE_BLOCKS = {
    "early": (61, 10_000),  # more than ~2 months before quarter end
    "mid": (1, 60),  # inside the quarter, up to ~2 months out
    "late": (-10_000, 0),  # quarter over, before/around the release
}

EPISODES = {
    "gfc": ("2008-07-01", "2009-12-31"),
    "covid": ("2020-01-01", "2021-06-30"),
}


def rmse(e: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.asarray(e, dtype=float) ** 2)))


def mae(e: pd.Series) -> float:
    return float(np.mean(np.abs(np.asarray(e, dtype=float))))


def load_predictions(
    results_dir: str | Path = "results",
    include_ablations: bool = False,
) -> pd.DataFrame:
    """Concatenate every ``preds/{model}.parquet`` under ``results_dir``.

    The ablation runs write ``{model}__{ablation}.parquet`` carrying the SAME
    ``model`` value and a different ``variant``, so anything selecting an arm
    by ``model`` alone silently pools the real run with the deliberately leaky
    ex-post panel. That pooling lowered the factor model's relative RMSE from
    0.700 to 0.631 in one robustness table and raised its quarter count from
    78 to 81, because the ablation answers quarters the real run abstains on.
    Ablation rows are therefore dropped unless asked for.
    """
    preds_dir = Path(results_dir) / "preds"
    files = sorted(preds_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no prediction files under {preds_dir}")
    frames = [pd.read_parquet(f) for f in files]
    out = pd.concat(frames, ignore_index=True)
    if not include_ablations and "variant" in out.columns:
        keep = ~out["variant"].astype(str).str.contains("\\+", regex=True)
        out = out[keep].reset_index(drop=True)
    out["nowcast_date"] = pd.to_datetime(out["nowcast_date"])
    out["target_quarter"] = pd.to_datetime(out["target_quarter"])
    return out


def load_target(
    path: str | Path = "data/processed/target_second_release.csv",
    growth_formula: str = "compounded",
) -> pd.Series:
    """The second-release target (R2), as a quarter -> growth series."""
    df = pd.read_csv(path)
    col = f"growth_{growth_formula}"
    if col not in df.columns:
        raise ValueError(f"{path} has no column {col}; found {list(df.columns)}")
    s = pd.Series(df[col].to_numpy(), index=pd.to_datetime(df["quarter"]))
    return s.sort_index().rename("actual")


def cycle_block(horizon_days: float) -> str:
    for name, (lo, hi) in CYCLE_BLOCKS.items():
        if lo <= horizon_days <= hi:
            return name
    return "late"


def prepare(
    preds: pd.DataFrame,
    target: pd.Series,
    genuine_only: bool = True,
) -> pd.DataFrame:
    """Join predictions to the target and attach error/episode/block columns."""
    df = preds.copy()
    df["actual"] = df["target_quarter"].map(target)
    if genuine_only and "target_already_published" in df.columns:
        df = df[~df["target_already_published"].astype(bool)]
    df = df[df["actual"].notna()].copy()
    df["error"] = df["yhat"] - df["actual"]
    df["block"] = df["horizon_days"].map(cycle_block)
    df["episode"] = "normal"
    for name, (lo, hi) in EPISODES.items():
        mask = (df["target_quarter"] >= pd.Timestamp(lo)) & (
            df["target_quarter"] <= pd.Timestamp(hi)
        )
        df.loc[mask, "episode"] = name
    return df


def coverage_report(prepared: pd.DataFrame) -> pd.DataFrame:
    """Attempted vs answered per model -- abstentions must stay visible."""
    g = prepared.groupby("model")
    out = pd.DataFrame(
        {
            "n_pairs": g.size(),
            "n_answered": g["yhat"].apply(lambda s: int(s.notna().sum())),
        }
    )
    out["coverage"] = (out["n_answered"] / out["n_pairs"]).round(4)
    return out.sort_index()


def common_sample(prepared: pd.DataFrame) -> pd.DataFrame:
    """Restrict to (date, quarter) pairs every model answered.

    Without this, a model that abstains on hard dates is rewarded for it, and
    an RMSE gap can reflect differing samples rather than differing accuracy.
    """
    answered = prepared[prepared["yhat"].notna()]
    keys = ["nowcast_date", "target_quarter"]
    counts = answered.groupby(keys)["model"].nunique()
    n_models = answered["model"].nunique()
    keep = counts[counts == n_models].index
    if len(keep) == 0:
        logger.warning("no (date, quarter) pair is covered by all models")
        return answered.iloc[0:0]
    return answered.set_index(keys).loc[keep].reset_index()


def error_table(
    preds: pd.DataFrame,
    target: pd.Series,
    benchmark: str = "dfm",
    restrict_common: bool = True,
) -> pd.DataFrame:
    """RMSE/MAE by model x cycle block x episode (paper Table T1).

    ``rmse_rel_benchmark`` is the ratio to ``benchmark`` within the same
    block/episode cell, so <1 means better than the DFM there.
    """
    prepared = prepare(preds, target)
    if restrict_common:
        prepared = common_sample(prepared)
    if prepared.empty:
        return pd.DataFrame()

    rows = []
    for (model, block, episode), g in prepared.groupby(["model", "block", "episode"]):
        rows.append(
            {
                "model": model,
                "block": block,
                "episode": episode,
                "n": len(g),
                "rmse": rmse(g["error"]),
                "mae": mae(g["error"]),
                "bias": float(g["error"].mean()),
            }
        )
    # Aggregate over blocks/episodes too ("all" rows).
    for (model,), g in prepared.groupby(["model"]):
        rows.append(
            {
                "model": model,
                "block": "all",
                "episode": "all",
                "n": len(g),
                "rmse": rmse(g["error"]),
                "mae": mae(g["error"]),
                "bias": float(g["error"].mean()),
            }
        )

    out = pd.DataFrame(rows)
    bench = (
        out[out["model"] == benchmark].set_index(["block", "episode"])["rmse"].to_dict()
    )
    out["rmse_rel_benchmark"] = [
        (
            (r["rmse"] / bench[(r["block"], r["episode"])])
            if (r["block"], r["episode"]) in bench
            and bench[(r["block"], r["episode"])] > 0
            else np.nan
        )
        for _, r in out.iterrows()
    ]
    return out.sort_values(["episode", "block", "rmse"]).reset_index(drop=True)


def cycle_profile(preds: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """RMSE by model and week of the publication cycle (RQ2; Figure F2)."""
    prepared = common_sample(prepare(preds, target))
    if prepared.empty:
        return pd.DataFrame()
    prepared = prepared.copy()
    # Week 0 = quarter end; positive = weeks BEFORE quarter end.
    prepared["cycle_week"] = np.ceil(prepared["horizon_days"] / 7).astype(int)
    rows = []
    for (model, week), g in prepared.groupby(["model", "cycle_week"]):
        rows.append(
            {
                "model": model,
                "cycle_week": int(week),
                "n": len(g),
                "rmse": rmse(g["error"]),
                "mae": mae(g["error"]),
            }
        )
    return (
        pd.DataFrame(rows).sort_values(["model", "cycle_week"]).reset_index(drop=True)
    )
