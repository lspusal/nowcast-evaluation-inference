"""External validation of our DFM against the NY Fed Staff Nowcast (M1).

Milestone M1 acceptance (docs/00_master_plan.md): correlate our own DFM
nowcasts with the published NY Fed Staff Nowcast over the overlapping
period, target rho >= 0.8, and document deviations.

Two properties of the published series shape how this comparison must be
read, and both are carried through into the output rather than averaged away:

* REAL-TIME vs RECONSTRUCTED. The NY Fed's own file states that forecasts
  for 2016Q1 onward "were made in real-time", while 2002Q1-2015Q4 are
  "historical reconstructions based on real-time data". Only the former are
  a true out-of-sample benchmark, so the correlation is reported separately
  for each regime instead of pooled.
* TWO MODEL GENERATIONS. Publication was suspended in September 2021 and
  relaunched in September 2023 as "Nowcast 2.0", with time-varying
  parameters and non-normal (outlier-robust) innovations. Those are
  different models, so the post-2023 overlap is reported as its own regime.

Sources (public):
  https://www.newyorkfed.org/research/policy/nowcast
  archive:  .../nowcast/New-York-Fed-Staff-Nowcast_data_2002-2021.xlsx
  current:  .../NowCast/Downloads/New-York-Fed-Staff-Nowcast_download_data.xlsx

Raw workbooks are cached under ``data/raw/nyfed/`` (gitignored): the repo
links to the source rather than redistributing it, per docs/01_data_spec.md §1.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NYFED_DIR = Path("data/raw/nyfed")
SHEET = "Forecasts By Quarter"

# The NY Fed's own annotation of its archive file.
REALTIME_FROM = pd.Timestamp("2016-01-01")
# Publication suspended 2021-09; relaunched 2023-09 as a different model.
SUSPENDED_FROM = pd.Timestamp("2021-09-30")
RELAUNCH_FROM = pd.Timestamp("2023-09-01")


def _read_by_quarter(path: Path) -> pd.DataFrame:
    """Long-form (forecast_date, target_quarter, nyfed) from one workbook."""
    raw = pd.read_excel(path, sheet_name=SHEET, header=None)
    header_idx = None
    for i in range(min(40, len(raw))):
        if (
            raw.iloc[i]
            .astype(str)
            .str.contains("Forecast Date", case=False, na=False)
            .any()
        ):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"no 'Forecast Date' header found in {path}")

    header = raw.iloc[header_idx]
    body = raw.iloc[header_idx + 1 :].copy()
    body.columns = header.to_numpy()
    date_col = header.iloc[0]
    body = body.rename(columns={date_col: "forecast_date"})
    body = body[body["forecast_date"].notna()]
    body["forecast_date"] = pd.to_datetime(body["forecast_date"], errors="coerce")
    body = body[body["forecast_date"].notna()]

    quarter_cols = [
        c for c in body.columns if isinstance(c, str) and len(c) == 6 and "Q" in c
    ]
    long = body.melt(
        id_vars="forecast_date",
        value_vars=quarter_cols,
        var_name="quarter_label",
        value_name="nyfed",
    ).dropna(subset=["nyfed"])
    long["target_quarter"] = pd.PeriodIndex(
        long["quarter_label"], freq="Q"
    ).to_timestamp(how="start")
    long["nyfed"] = pd.to_numeric(long["nyfed"], errors="coerce")
    return long.dropna(subset=["nyfed"])[["forecast_date", "target_quarter", "nyfed"]]


def load_nyfed_nowcast(nyfed_dir: str | Path = NYFED_DIR) -> pd.DataFrame:
    """All published NY Fed Staff Nowcast vintages, both model generations."""
    nyfed_dir = Path(nyfed_dir)
    frames = []
    for name in ("nyfed_2002_2021", "nyfed_current"):
        path = nyfed_dir / f"{name}.xlsx"
        if not path.exists():
            logger.warning("missing NY Fed workbook %s", path)
            continue
        df = _read_by_quarter(path)
        df["source"] = name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no NY Fed workbooks under {nyfed_dir}")

    out = pd.concat(frames, ignore_index=True)
    # Prefer the archive where both cover a date (it is the published
    # real-time record for that era).
    out = out.sort_values(["target_quarter", "forecast_date", "source"])
    out = out.drop_duplicates(["forecast_date", "target_quarter"], keep="first")

    out["regime"] = np.where(
        out["forecast_date"] < REALTIME_FROM,
        "reconstructed",
        np.where(out["forecast_date"] >= RELAUNCH_FROM, "realtime_v2", "realtime_v1"),
    )
    return out.reset_index(drop=True)


def compare_dfm_to_nyfed(
    preds: pd.DataFrame,
    nyfed: pd.DataFrame,
    model: str = "dfm",
    tolerance_days: int = 4,
) -> pd.DataFrame:
    """Match our nowcasts to NY Fed's by (target_quarter, nearest date).

    The NY Fed publishes on Fridays; our calendar is weekly but anchored on
    the quarter end, so dates rarely coincide exactly. Each of our
    (date, quarter) pairs is matched to the NY Fed vintage for the SAME
    target quarter whose forecast date is nearest and within
    ``tolerance_days`` -- never to a LATER NY Fed vintage beyond that window,
    which would compare our nowcast against a better-informed one.
    """
    ours = preds[(preds["model"] == model) & preds["yhat"].notna()].copy()
    if ours.empty:
        return pd.DataFrame()
    ours["nowcast_date"] = pd.to_datetime(ours["nowcast_date"])
    ours["target_quarter"] = pd.to_datetime(ours["target_quarter"])

    rows = []
    for quarter, g in ours.groupby("target_quarter"):
        ref = nyfed[nyfed["target_quarter"] == quarter]
        if ref.empty:
            continue
        ref = ref.sort_values("forecast_date")
        merged = pd.merge_asof(
            g.sort_values("nowcast_date"),
            ref[["forecast_date", "nyfed", "regime"]],
            left_on="nowcast_date",
            right_on="forecast_date",
            direction="nearest",
            tolerance=pd.Timedelta(days=tolerance_days),
        )
        rows.append(merged)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True).dropna(subset=["nyfed"])
    out["gap_days"] = (out["nowcast_date"] - out["forecast_date"]).dt.days
    return out[
        [
            "nowcast_date",
            "target_quarter",
            "yhat",
            "nyfed",
            "regime",
            "forecast_date",
            "gap_days",
            "horizon_days",
        ]
    ]


def nyfed_validation_table(matched: pd.DataFrame) -> pd.DataFrame:
    """Correlation of our DFM with the NY Fed nowcast, by regime (M1 accept.).

    Reported per regime because only ``realtime_v1``/``realtime_v2`` are
    genuine published real-time nowcasts; ``reconstructed`` values were
    produced after the fact by the NY Fed itself.
    """
    if matched.empty:
        return pd.DataFrame()

    def _stats(g: pd.DataFrame, label: str) -> dict:
        pearson = g["yhat"].corr(g["nyfed"])
        spearman = g["yhat"].corr(g["nyfed"], method="spearman")
        return {
            "regime": label,
            "n": len(g),
            "first_date": g["nowcast_date"].min().date(),
            "last_date": g["nowcast_date"].max().date(),
            "rho_pearson": round(float(pearson), 4) if pd.notna(pearson) else np.nan,
            "rho_spearman": round(float(spearman), 4) if pd.notna(spearman) else np.nan,
            "mean_abs_diff": round(float((g["yhat"] - g["nyfed"]).abs().mean()), 4),
            "our_mean": round(float(g["yhat"].mean()), 4),
            "nyfed_mean": round(float(g["nyfed"].mean()), 4),
            "meets_target_rho_0.8": bool(pd.notna(pearson) and pearson >= 0.8),
        }

    rows = [_stats(g, regime) for regime, g in matched.groupby("regime")]
    realtime = matched[matched["regime"].isin(["realtime_v1", "realtime_v2"])]
    if not realtime.empty:
        rows.append(_stats(realtime, "realtime_all"))
    rows.append(_stats(matched, "all"))
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)
