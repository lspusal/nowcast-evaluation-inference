"""Experiment orchestrator (pseudo-real-time backtest).

Implements docs/02_methodology.md §2 and the Makefile stages:
    --stage download  : M0 data pipeline (alfred_client + info_set + target)
    --stage all       : full backtest for all configured models
    --stage evaluate  : metrics + DM + MCS from saved predictions
    --stage tables    : paper tables/figures (T1-T7, F1-F5) from results

Run manifest (R4): results/<run_id>/manifest.json with config hash,
package versions, timestamps, per-model compute stats.

DONE: CLI + config + seeding (R4); stage 'download' (M0) writing the
coverage table and the second-release target, with a run manifest carrying
the config, package versions and timings. Stage 'all' for the econometric
family (M1): information set built ONCE per nowcast date and reused across
models (R3), refit cadence honoured, checkpointed to parquet.

Stage 'evaluate' runs the src/eval pipeline into results/eval/*.csv; stage
'tables' generates every table in paper/outline.md from results/ (figures are
built separately by src/viz/figures.py); '--ablation {ragged_edge,
balanced_panel}' runs the M5 ablations. Hand-edited numbers in any of these
outputs are prohibited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import alfred_client, info_set
from src.data.transforms import transform_panel
from src.models.econ_baselines import ECON_MODELS
from src.models.ml_baselines import ML_MODELS
from src.models.tsfm_models import (
    TSFM_MODELS,
    ChronosFineTunedNowcaster,
    ChronosFineTuner,
)

# Model registry across families, with the config key that carries each
# family's refit cadence (docs/02_methodology.md §2.3).
_CONFIG_KEYS = {"tsfm": "tsfm_zero_shot"}

MODEL_FAMILIES = {
    "econ": ECON_MODELS,
    "ml": ML_MODELS,
    "tsfm": TSFM_MODELS,
}

logger = logging.getLogger(__name__)

# Predictions are flushed to parquet this often. Small enough that an
# interruption (sleep, shutdown, closed terminal) costs only a few minutes of
# the slowest model's time; large enough that parquet rewriting is not the
# bottleneck.
FLUSH_EVERY = 10

# The common prediction schema (docs/02_methodology.md §2.4).
PRED_COLUMNS = [
    "nowcast_date",
    "target_quarter",
    "yhat",
    "model",
    "variant",
    "run_id",
    "horizon_days",
    "target_already_published",
    # Which model code produced the row. A resumed run must not inherit rows
    # written by code that has since been corrected; see _load_done_pairs.
    "code_fingerprint",
]


def code_fingerprint() -> str:
    """A short hash of the source that determines a prediction.

    The checkpoint written by a backtest carries no record of the code that
    produced it, so a run resumed after a model fix silently keeps the old
    rows. That happened: an arm whose panel completer was returning an empty
    context wrote 1,100 abstentions, the fix was applied, and the resumed run
    treated those 1,100 dates as already done and recomputed only the rest.
    The output looked complete and was 78% stale.
    """
    h = hashlib.sha256()
    paths = sorted((Path(__file__).resolve().parents[1] / "models").glob("*.py"))
    paths.append(Path(__file__).resolve().parents[1] / "data" / "info_set.py")
    for f in paths:
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def set_global_seed(cfg: dict) -> None:
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)


def run_stage_download(cfg: dict, panel: str = "all") -> dict:
    """M0 data pipeline: download vintages, build target, write coverage table."""
    alfred_client.download_panel(cfg, panel=panel)
    # strict=False: report coverage even when a tier is deliberately partial
    # (e.g. --panel baseline); completeness is visible in the `complete` column.
    coverage = alfred_client.coverage_table(cfg, strict=(panel == "all"))
    results_dir = Path(cfg.get("output_dir", "results"))
    coverage_path = results_dir / "m0_data_coverage.csv"
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(coverage_path, index=False)

    target_id = cfg.get("target", {}).get("fred_id", "GDPC1")
    processed_dir = Path(cfg.get("processed_dir", "data/processed"))
    target_df = info_set.save_second_release_target(
        target_id,
        out_path=str(processed_dir / "target_second_release.csv"),
        gap_min_days=cfg.get("target", {}).get(
            "release_gap_min_days", info_set.RELEASE_GAP_MIN_DAYS
        ),
        gap_max_days=cfg.get("target", {}).get(
            "release_gap_max_days", info_set.RELEASE_GAP_MAX_DAYS
        ),
    )

    return {
        "n_series": len(coverage),
        "coverage_path": str(coverage_path),
        "n_target_quarters": len(target_df),
        "target_first_quarter": (
            str(target_df["quarter"].min()) if len(target_df) else None
        ),
        "target_last_quarter": (
            str(target_df["quarter"].max()) if len(target_df) else None
        ),
        "growth_formula": cfg.get("target", {}).get(
            "growth_formula", info_set.DEFAULT_GROWTH_FORMULA
        ),
        "panel_baseline": alfred_client.panel_series_ids(cfg, "baseline"),
        "panel_extended": [
            s
            for s in alfred_client.panel_series_ids(cfg, "all")
            if s not in alfred_client.panel_series_ids(cfg, "baseline")
        ],
    }


def _load_done_pairs(path: Path, fingerprint: str | None = None) -> set[tuple]:
    """Already-computed (nowcast_date, target_quarter) pairs, for resuming.

    Rows written by different model code are not reusable, so when
    ``fingerprint`` is given they are dropped and recomputed. A checkpoint
    predating the fingerprint column carries no such record; those rows are
    kept, with a warning, because refusing them would discard every result
    produced before the column existed.
    """
    if not path.exists():
        return set()
    try:
        done = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 - a corrupt checkpoint must not be fatal
        logger.warning("could not read checkpoint %s (%s); recomputing", path, exc)
        return set()
    if fingerprint is not None:
        if "code_fingerprint" not in done.columns:
            logger.warning(
                "checkpoint %s predates the code fingerprint; its %d rows are "
                "being reused without being able to check which code wrote them",
                path,
                len(done),
            )
        else:
            stale = done["code_fingerprint"].ne(fingerprint)
            if stale.any():
                logger.warning(
                    "checkpoint %s: discarding %d of %d rows written by other "
                    "model code (%s); they will be recomputed",
                    path,
                    int(stale.sum()),
                    len(done),
                    ", ".join(sorted(done.loc[stale, "code_fingerprint"].unique())),
                )
                done = done[~stale]
    return {
        (pd.Timestamp(a).date(), pd.Timestamp(b))
        for a, b in zip(done["nowcast_date"], done["target_quarter"])
    }


def run_stage_all(
    cfg: dict,
    run_id: str,
    panel: str = "baseline",
    families: tuple[str, ...] | None = None,
    ablation: str | None = None,
) -> dict:
    """Pseudo-real-time backtest for the econometric family (M1).

    docs/02_methodology.md §2: for each nowcast date the information set is
    built ONCE and reused across models (R3) -- both to keep every model on
    literally the same information and because rebuilding it per model is the
    expensive part. Models are refitted every
    ``refit_every_quarters.econ`` quarters; between refits the fitted objects
    are reused, which is what makes a 1,400-date backtest tractable.

    Predictions are appended per model to ``results/preds/{model}.parquet``
    with the common schema, checkpointed so a crashed run resumes without
    recomputing finished dates.
    """
    data_cfg = load_config(cfg.get("data_config", "configs/data.yaml"))
    series = alfred_client.panel_series_ids(data_cfg, panel)
    target_id = data_cfg["target"]["fred_id"]
    growth_formula = data_cfg["target"].get("growth_formula")
    gap_min = data_cfg["target"].get(
        "release_gap_min_days", info_set.RELEASE_GAP_MIN_DAYS
    )
    gap_max = data_cfg["target"].get(
        "release_gap_max_days", info_set.RELEASE_GAP_MAX_DAYS
    )

    sample = data_cfg["sample"]
    weeks = cfg.get("nowcast", {}).get("weeks_before_release", 13)
    pairs = info_set.nowcast_calendar(
        sample["nowcast_start"],
        sample["nowcast_end"],
        weeks_before_quarter_end=weeks,
        series_id=target_id,
    )
    # Resolve which models to run, from which family, with which params.
    # A family's entry may be a list (econ) or a mapping of model ->
    # hyperparameters (ml); both shapes appear in configs/experiment_*.yaml.
    cadences = cfg.get("refit_every_quarters", {})
    specs: dict[str, dict] = {}
    for family, registry in MODEL_FAMILIES.items():
        requested = cfg.get("models", {}).get(_CONFIG_KEYS.get(family, family)) or []
        params_by_model = requested if isinstance(requested, dict) else {}
        names = list(requested)
        for name in names:
            if name not in registry:
                logger.warning("skipping unknown %s model %r", family, name)
                continue
            if families and family not in families:
                continue
            specs[name] = {
                "cls": registry[name],
                "family": family,
                "params": dict(params_by_model.get(name) or {}),
                "refit_every": cadences.get(family, 1),
            }
    ft_budgets = [b for b in cfg.get("finetune_budget_steps", []) if b > 0]
    ft_mode = cfg.get("finetune_mode", "lora")
    ft_tag = "lora_ft" if ft_mode == "lora" else "full_ft"
    run_ft = bool(ft_budgets) and (not families or "tsfm_ft" in families)
    if run_ft:
        for b in ft_budgets:
            specs[f"chronos_{ft_tag}{b}"] = {
                "cls": None,  # built per refit from the shared tuner
                "family": "tsfm_ft",
                "params": {"budget": b},
                "refit_every": cadences.get("tsfm_ft", 4),
            }

    if not specs:
        raise ValueError("no known models configured under models.*")
    model_names = list(specs)

    suffix = f"__{ablation}" if ablation else ""
    preds_dir = Path(cfg.get("output_dir", "results")) / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = code_fingerprint()
    logger.info("model code fingerprint %s", fingerprint)
    done = {
        name: _load_done_pairs(preds_dir / f"{name}{suffix}.parquet", fingerprint)
        for name in model_names
    }

    fitted: dict[str, object] = {}
    last_fit_quarter: dict[str, pd.Timestamp] = {}
    rows: dict[str, list[dict]] = {name: [] for name in model_names}
    n_done = n_skipped = 0
    t_start = time.time()

    for nowcast_date, target_quarter in pairs:
        pending = [
            n for n in model_names if (nowcast_date, target_quarter) not in done[n]
        ]
        if not pending:
            n_skipped += 1
            continue

        # Built ONCE, reused by every model (R3).
        if ablation == "balanced_panel":
            # H3 ablation: deliberately leaky ex-post panel, upper bound only.
            panel_df = info_set.balanced_panel(
                series, target_quarter, target_id=target_id
            )
        else:
            panel_df = info_set.information_set(
                nowcast_date, series, target_id=target_id, target_quarter=target_quarter
            )
        X = transform_panel(panel_df, data_cfg)
        y = info_set.available_target(
            nowcast_date,
            target_id,
            growth_formula=growth_formula,
            gap_min_days=gap_min,
            gap_max_days=gap_max,
        )
        published = info_set.target_already_published(
            nowcast_date, target_quarter, target_id
        )

        # Refit per family cadence (econ every quarter, ml every 4 by
        # default); reuse the fitted objects in between, which is what makes a
        # 1,400-date backtest tractable.
        # One LoRA run per refit serves every budget (nested sweep).
        if run_ft:
            ft_last = last_fit_quarter.get("__ft__")
            ft_cadence = cadences.get("tsfm_ft", 4)
            if ft_last is None or (
                (target_quarter.to_period("Q") - ft_last.to_period("Q")).n >= ft_cadence
            ):
                tuner = ChronosFineTuner(
                    budgets=tuple(ft_budgets),
                    seed=cfg.get("seed", 42),
                    mode=ft_mode,
                )
                try:
                    tuner.fit(X, seed=cfg.get("seed", 42))
                    shared_tuner = tuner
                except Exception as exc:  # noqa: BLE001
                    logger.warning("fine-tuning failed at %s (%s)", nowcast_date, exc)
                    shared_tuner = None
                last_fit_quarter["__ft__"] = target_quarter
                for b in ft_budgets:
                    m = ChronosFineTunedNowcaster(budget=b, shared_tuner=shared_tuner)
                    try:
                        m.fit(X, y)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "ft%d fit failed at %s (%s)", b, nowcast_date, exc
                        )
                    fitted[f"chronos_{ft_tag}{b}"] = m
                    last_fit_quarter[f"chronos_{ft_tag}{b}"] = target_quarter

        for name, spec in specs.items():
            if spec["cls"] is None:
                continue  # fine-tuned models are constructed above
            last_q = last_fit_quarter.get(name)
            need_refit = last_q is None or (
                (target_quarter.to_period("Q") - last_q.to_period("Q")).n
                >= spec["refit_every"]
            )
            if not need_refit:
                continue
            kwargs = dict(spec["params"])
            if name == "dfm":
                kwargs.setdefault("start", sample.get("train_start"))
            if spec["family"] == "ml":
                kwargs.setdefault("seed", cfg.get("seed", 42))
            override = cfg.get("min_train_quarters_override")
            if override is not None:
                # Short samples (the euro area has 41 target quarters vs 138
                # for the US) cannot afford the default 20-quarter warm-up.
                kwargs.setdefault("min_train_quarters", override)
            if spec["family"] == "tsfm":
                # Zero-shot at M3: the foundation model is frozen (H1).
                kwargs.pop("modes", None)
                kwargs.pop("use_timegpt", None)
            model = _instantiate(spec["cls"], kwargs)
            try:
                model.fit(X, y)
            except Exception as exc:  # noqa: BLE001
                # One model failing must not abort a multi-hour backtest.
                logger.warning("%s: fit failed at %s (%s)", name, nowcast_date, exc)
            fitted[name] = model
            last_fit_quarter[name] = target_quarter

        horizon = (
            pd.Timestamp(target_quarter).to_period("Q").end_time
            - pd.Timestamp(nowcast_date)
        ).days
        for name in pending:
            model = fitted.get(name)
            yhat = float("nan")
            if model is None:
                # Silent until now, which is how a full-fine-tuning run wrote
                # 1,403 NaNs with a clean log.
                logger.warning(
                    "%s: no fitted model at %s (fitted keys: %s)",
                    name,
                    nowcast_date,
                    sorted(fitted),
                )
            if model is not None:
                try:
                    yhat = _predict(model, X, target_quarter, y)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "%s: predict failed at %s (%s)", name, nowcast_date, exc
                    )
            rows[name].append(
                {
                    "nowcast_date": pd.Timestamp(nowcast_date),
                    "target_quarter": target_quarter,
                    "yhat": yhat,
                    "model": name,
                    "variant": f"{panel}+{ablation}" if ablation else panel,
                    "run_id": run_id,
                    "horizon_days": horizon,
                    "target_already_published": published,
                    "code_fingerprint": fingerprint,
                }
            )
        n_done += 1

        # Flush periodically so a crash loses at most a few dates.
        if n_done % FLUSH_EVERY == 0:
            _flush_preds(preds_dir, rows, suffix)
            logger.info(
                "%d/%d dates (%.1f min elapsed)",
                n_done,
                len(pairs),
                (time.time() - t_start) / 60,
            )

    _flush_preds(preds_dir, rows, suffix)

    counts = {}
    for name in model_names:
        path = preds_dir / f"{name}{suffix}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            counts[name] = {
                "n_predictions": len(df),
                "n_finite": int(df["yhat"].notna().sum()),
            }
    return {
        "n_pairs": len(pairs),
        "n_computed": n_done,
        "n_skipped_checkpoint": n_skipped,
        "panel": panel,
        "models": counts,
        "duration_minutes": round((time.time() - t_start) / 60, 1),
    }


def _instantiate(cls, kwargs: dict):
    """Construct a model, dropping kwargs its signature does not accept.

    Config carries hyperparameters per model family; this keeps an unknown or
    extra key from crashing a multi-hour run while still passing through
    everything the model does understand.
    """
    import inspect

    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return cls()
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_kwargs:
        return cls(**kwargs)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    dropped = set(kwargs) - set(allowed)
    if dropped:
        logger.debug(
            "%s: ignoring unsupported params %s", cls.__name__, sorted(dropped)
        )
    return cls(**allowed)


def _predict(model, X, target_quarter, y):
    """Call predict, passing the training target when the model accepts it."""
    import inspect

    try:
        sig = inspect.signature(model.predict)
        if "y" in sig.parameters:
            return model.predict(X, target_quarter, y).mean
    except (TypeError, ValueError):
        pass
    return model.predict(X, target_quarter).mean


def _flush_preds(
    preds_dir: Path, rows: dict[str, list[dict]], suffix: str = ""
) -> None:
    """Append buffered rows to each model's parquet, then clear the buffer."""
    for name, buf in rows.items():
        if not buf:
            continue
        path = preds_dir / f"{name}{suffix}.parquet"
        new = pd.DataFrame(buf)[PRED_COLUMNS]
        if path.exists():
            new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
        new.to_parquet(path, index=False)
        buf.clear()


def run_stage_evaluate(cfg: dict) -> dict:
    """M6 evaluation pipeline over saved predictions -> results/eval/*.csv.

    Every number the study reports comes from here; nothing is transcribed by
    hand. The M1 external validation against the NY
    Fed Staff Nowcast is written as results/m1_dfm_vs_nyfed.csv, which is the
    milestone's acceptance artifact.
    """
    from src.eval import metrics as M
    from src.eval.dm_test import dm_table
    from src.eval.mcs import mcs_table
    from src.eval.nyfed import (
        compare_dfm_to_nyfed,
        load_nyfed_nowcast,
        nyfed_validation_table,
    )

    data_cfg = load_config(cfg.get("data_config", "configs/data.yaml"))
    out_dir = Path(cfg.get("output_dir", "results"))
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    preds = M.load_predictions(out_dir)
    target = M.load_target(
        Path(data_cfg.get("processed_dir", "data/processed"))
        / "target_second_release.csv",
        growth_formula=data_cfg["target"].get("growth_formula", "compounded"),
    )
    prepared = M.prepare(preds, target)
    common = M.common_sample(prepared)

    ev = cfg.get("evaluation", {})
    written = {}

    coverage = M.coverage_report(prepared)
    coverage.to_csv(eval_dir / "coverage.csv")
    written["coverage"] = str(eval_dir / "coverage.csv")

    errors = M.error_table(preds, target)
    errors.to_csv(eval_dir / "t1_error_table.csv", index=False)
    written["t1_error_table"] = str(eval_dir / "t1_error_table.csv")

    profile = M.cycle_profile(preds, target)
    profile.to_csv(eval_dir / "f2_cycle_profile.csv", index=False)
    written["f2_cycle_profile"] = str(eval_dir / "f2_cycle_profile.csv")

    # hac_lags omitted on purpose: dm_table derives it from the grid.
    dm = dm_table(common, benchmark="dfm", hac_lags=ev.get("dm_hac_lags"))
    dm.to_csv(eval_dir / "dm_vs_dfm.csv", index=False)
    written["dm_vs_dfm"] = str(eval_dir / "dm_vs_dfm.csv")

    try:
        conf_set = mcs_table(
            common, alpha=ev.get("mcs_alpha", 0.10), seed=cfg.get("seed", 42)
        )
        conf_set.to_csv(eval_dir / "t2_mcs.csv", index=False)
        written["t2_mcs"] = str(eval_dir / "t2_mcs.csv")
    except ValueError as exc:
        logger.warning("MCS skipped: %s", exc)

    nyfed_summary = None
    try:
        matched = compare_dfm_to_nyfed(preds, load_nyfed_nowcast())
        table = nyfed_validation_table(matched)
        table.to_csv(out_dir / "m1_dfm_vs_nyfed.csv", index=False)
        matched.to_csv(eval_dir / "m1_dfm_vs_nyfed_matched.csv", index=False)
        written["m1_dfm_vs_nyfed"] = str(out_dir / "m1_dfm_vs_nyfed.csv")
        row = table[table["regime"] == "realtime_v1"]
        if not row.empty:
            nyfed_summary = {
                "regime": "realtime_v1",
                "n": int(row.iloc[0]["n"]),
                "rho_pearson": float(row.iloc[0]["rho_pearson"]),
                "meets_target": bool(row.iloc[0]["meets_target_rho_0.8"]),
            }
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("NY Fed validation skipped: %s", exc)

    return {
        "n_predictions": len(preds),
        "n_genuine_forecast_pairs": len(prepared),
        "n_common_sample": len(common),
        "models": sorted(preds["model"].unique().tolist()),
        "artifacts": written,
        "nyfed_validation": nyfed_summary,
    }


def run_stage_tables(cfg: dict) -> dict:
    """M6: generate every paper table and figure from results/ (no hand numbers). forbids transcribing results into the study by hand, so every
    T*/F* artifact listed in paper/outline.md is produced here from the saved
    predictions. Anything that depends on a milestone not yet run (F3, the
    fine-tuning budget curve; T4, the euro area) is SKIPPED with a recorded
    reason rather than emitted empty -- a missing table is honest, an empty
    one invites being filled in by hand later.
    """
    from src.eval import metrics as M
    from src.eval.dm_test import dm_table
    from src.eval.mcs import mcs_table

    data_cfg = load_config(cfg.get("data_config", "configs/data.yaml"))
    out_dir = Path(cfg.get("output_dir", "results"))
    tab_dir = out_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    preds = M.load_predictions(out_dir)
    # Ablation runs are a separate experiment; they must not contaminate the
    # headline tables.
    preds = preds[~preds["variant"].astype(str).str.contains("balanced")]
    target = M.load_target(
        Path(data_cfg.get("processed_dir", "data/processed"))
        / "target_second_release.csv",
        growth_formula=data_cfg["target"].get("growth_formula", "compounded"),
    )
    prepared = M.prepare(preds, target)
    common = M.common_sample(prepared)
    ev = cfg.get("evaluation", {})
    written, skipped = {}, {}

    # T1 -- RMSE relative to the DFM by model and cycle block.
    t1 = M.error_table(preds, target)
    t1.to_csv(tab_dir / "T1_rmse_by_model_and_block.csv", index=False)
    written["T1"] = "RMSE/MAE by model x block x episode, relative to DFM"

    # T2 -- Model Confidence Set.
    try:
        t2 = mcs_table(
            common, alpha=ev.get("mcs_alpha", 0.10), seed=cfg.get("seed", 42)
        )
        t2.to_csv(tab_dir / "T2_model_confidence_set.csv", index=False)
        written["T2"] = "Model Confidence Set"
    except ValueError as exc:
        skipped["T2"] = str(exc)

    # T3 -- crisis episodes.
    t3 = t1[t1["episode"].isin(["gfc", "covid"])]
    t3.to_csv(tab_dir / "T3_crisis_episodes.csv", index=False)
    written["T3"] = "RMSE by model within the 2008 and 2020 episodes"

    # T4 -- euro-area extension (M5 / RQ4). Built from the intl run if present.
    intl_dir = Path(cfg.get("intl_output_dir", "results_intl12"))
    intl_target = Path("data/processed_intl/target_second_release.csv")
    if (intl_dir / "preds").exists() and intl_target.exists():
        intl_preds = M.load_predictions(intl_dir)
        intl_y = M.load_target(intl_target)
        intl_prep = M.prepare(intl_preds, intl_y)
        # Pairwise-vs-DFM, because the all-model intersection collapses to
        # single digits on 41 quarters.
        keys = ["nowcast_date", "target_quarter"]
        answered = intl_prep[intl_prep["yhat"].notna()]
        wide = answered.pivot_table(index=keys, columns="model", values="error")
        rows = []
        if "dfm" in wide.columns:
            for model in wide.columns:
                if model == "dfm":
                    continue
                pair = wide[["dfm", model]].dropna()
                if len(pair) < 20:
                    continue
                covid = pair.index.get_level_values("target_quarter")
                is_covid = (covid >= pd.Timestamp("2020-01-01")) & (
                    covid <= pd.Timestamp("2021-06-30")
                )
                for label, sel in (("all", slice(None)), ("ex_covid", ~is_covid)):
                    g = pair[sel] if label == "ex_covid" else pair
                    if len(g) < 20:
                        continue
                    rows.append(
                        {
                            "subset": label,
                            "model": model,
                            "n_dates": len(g),
                            "n_quarters": g.index.get_level_values(
                                "target_quarter"
                            ).nunique(),
                            "rmse_dfm": float(np.sqrt((g["dfm"] ** 2).mean())),
                            "rmse_model": float(np.sqrt((g[model] ** 2).mean())),
                        }
                    )
        if rows:
            t4 = pd.DataFrame(rows)
            t4["rmse_rel_dfm"] = t4["rmse_model"] / t4["rmse_dfm"]
            # The DFM is not reliably estimable on this sample -- see §48. The
            # flag keeps that caveat attached to the numbers themselves.
            t4["caveat"] = "euro-area DFM unreliable on 41q post-COVID"
            t4.sort_values(["subset", "rmse_rel_dfm"]).to_csv(
                tab_dir / "T4_euro_area.csv", index=False
            )
            written["T4"] = "euro-area pairwise-vs-DFM (RQ4; see the design notes-49)"
        else:
            skipped["T4"] = "euro-area sample too small for any pairwise comparison"
    else:
        skipped["T4"] = f"no euro-area run found under {intl_dir}"

    # T5 -- ragged-edge ablation (H3), written by the ablation run.
    if (out_dir / "eval" / "t5_ragged_edge_ablation.csv").exists():
        import shutil

        shutil.copy(
            out_dir / "eval" / "t5_ragged_edge_ablation.csv",
            tab_dir / "T5_ragged_edge_ablation.csv",
        )
        written["T5"] = "ragged edge vs ex-post balanced panel (H3)"
    else:
        skipped["T5"] = "ablation run not found; run --ablation balanced_panel"

    # T6 -- variance decomposition of squared error. Reported in BOTH an
    # additive and a multiplicative (log) specification: the additive fit is
    # dominated by the few enormous COVID errors and understates the cycle-week
    # effect by more than an order of magnitude.
    dec = prepared[prepared["yhat"].notna()].copy()
    if not dec.empty and dec["model"].nunique() > 1:
        dec["sq"] = dec["error"] ** 2
        # Week 0 = quarter end; positive = weeks BEFORE quarter end (as in F2).
        dec["cycle_week"] = np.ceil(dec["horizon_days"] / 7).astype(int)
        # Main effects, plus the week x model INTERACTION. The interaction is
        # not optional here: the value of another week of data has opposite
        # signs across models (+47% for the DFM, -17% for a univariate TSFM,
        # exactly 0 for the naive mean), so a main-effects-only fit averages
        # the week effect to almost nothing and understates it.
        keys = {
            "quarter": ["target_quarter"],
            "quarter+week": ["target_quarter", "cycle_week"],
            "quarter+week+model": ["target_quarter", "cycle_week", "model"],
            "quarter+week*model": ["target_quarter", "cycle_week:model"],
        }
        dec["cycle_week:model"] = (
            dec["cycle_week"].astype(str) + ":" + dec["model"].astype(str)
        )
        rows = []
        for spec, response in (
            ("additive", dec["sq"].to_numpy(float)),
            # +eps guards the exactly-zero errors; log turns the fixed effects
            # multiplicative, which is the natural scale for a squared error.
            ("multiplicative", np.log(dec["sq"].to_numpy(float) + 1e-12)),
        ):
            tss = float(((response - response.mean()) ** 2).sum())
            for term, cols in keys.items():
                design = pd.get_dummies(
                    dec[cols].astype(str).agg("|".join, axis=1), dtype=float
                )
                if len(cols) > 1:
                    # main effects, not the saturated interaction: the
                    # interaction would fit every cell exactly (R2 = 1).
                    design = pd.concat(
                        [
                            pd.get_dummies(dec[c].astype(str), prefix=c, dtype=float)
                            for c in cols
                        ],
                        axis=1,
                    )
                x = np.column_stack([np.ones(len(design)), design.to_numpy()])
                coef, *_ = np.linalg.lstsq(x, response, rcond=None)
                rss = float(((response - x @ coef) ** 2).sum())
                rows.append(
                    {
                        "spec": spec,
                        "term": term,
                        "n_params": x.shape[1],
                        "r2": 1.0 - rss / tss if tss > 0 else float("nan"),
                    }
                )
        t6 = pd.DataFrame(rows)
        t6.to_csv(tab_dir / "T6_variance_decomposition.csv", index=False)
        written["T6"] = "variance decomposition of squared error (quarter/week/model)"
    else:
        skipped["T6"] = "need >1 model with answered nowcasts"

    # T7 -- every model against the TRIVIAL benchmark. Without this the study
    # can only say "no model beats the DFM", which is uninterpretable: it does
    # not distinguish a strong field from a field where nothing works at all
    #. Pairwise, because models abstain on
    # different dates and the all-model intersection is much smaller.
    from src.eval.cluster_inference import ibragimov_muller
    from src.eval.dm_test import dm_test, dm_test_by_quarter, hac_sensitivity

    naive_ref = ev.get("naive_benchmark", "naive_mean")
    answered_all = prepared[prepared["yhat"].notna()]
    wide_all = answered_all.pivot_table(
        index=["nowcast_date", "target_quarter"], columns="model", values="error"
    )
    if naive_ref in wide_all.columns:
        # The weekly HAC statistic is the audit trail, so it is computed at
        # the bandwidth this study originally used -- that IS the artifact
        # being audited (§66). It is named `*_weekly_hac` and never decides a
        # verdict.
        hac = ev.get("dm_hac_lags", 4)
        rows = []
        for model in wide_all.columns:
            if model == naive_ref:
                continue
            pair = wide_all[[naive_ref, model]].dropna()
            if len(pair) < ev.get("min_pairs_vs_naive", 50):
                continue
            rmse_n = float(np.sqrt((pair[naive_ref] ** 2).mean()))
            rmse_m = float(np.sqrt((pair[model] ** 2).mean()))
            # The weekly HAC statistic is kept only to document how much
            # significance the bandwidth manufactures; the REPORTED inference
            # is quarterly, where each target contributes one observation
            #.
            stat, pval = dm_test(pair[model], pair[naive_ref], hac_lags=hac)
            q = dm_test_by_quarter(pair[model], pair[naive_ref])
            # Two estimands, two verdicts. MEAN LOSS is tested by the wild
            # cluster bootstrap, which is the primary procedure: correctly
            # sized on this design and on real differentials, and it uses the
            # magnitude of the differential. THE TYPICAL QUARTER is tested by
            # the exact sign test, which answers a different question and is
            # reported as complementary rather than as a weaker version of
            # the same one.
            from src.eval.cluster_inference import wild_cluster_bootstrap

            d_pair = (pair[model] ** 2 - pair[naive_ref] ** 2).to_numpy(float)
            g_pair = pd.factorize(pair.index.get_level_values("target_quarter"))[0]
            p_wild = wild_cluster_bootstrap(
                d_pair,
                g_pair,
                n_boot=int(ev.get("bootstrap_draws", 1999)),
                seed=cfg.get("seed", 42),
            )["p"]
            # The study's PRIMARY procedure for mean loss. It was computed in
            # the inference audit but not carried into this table, so the
            # table reported a plain quarter-level t-test beside a bootstrap
            # the study declines to rely on, and neither was the test the
            # text says decides.
            p_im = ibragimov_muller(d_pair, g_pair, q=int(ev.get("im_groups", 4)))["p"]
            if p_wild < 0.05:
                verdict_mean = (
                    "lower mean loss" if rmse_m < rmse_n else "higher mean loss"
                )
            else:
                verdict_mean = "indistinguishable"
            if q["p_sign"] < 0.05:
                ahead = q["n_quarters_model_ahead"] > q["n_quarters"] / 2
                verdict_quarter = (
                    "ahead in most quarters" if ahead else "behind in most quarters"
                )
            else:
                verdict_quarter = "indistinguishable"
            rows.append(
                {
                    "model": model,
                    "benchmark": naive_ref,
                    "n": len(pair),
                    "n_quarters": q["n_quarters"],
                    "rmse_model": rmse_m,
                    "rmse_naive": rmse_n,
                    "rmse_rel_naive": rmse_m / rmse_n,
                    "t_stat_quarterly": q["t_stat"],
                    "p_im_q4": p_im,
                    "p_wild_bootstrap": p_wild,
                    "p_t_quarterly": q["p_t"],
                    "quarters_ahead": q["n_quarters_model_ahead"],
                    "p_sign": q["p_sign"],
                    "p_wilcoxon": q["p_wilcoxon"],
                    "dm_stat_weekly_hac": stat,
                    "p_weekly_hac": pval,
                    "verdict_mean_loss": verdict_mean,
                    "verdict_typical_quarter": verdict_quarter,
                }
            )
        if rows:
            pd.DataFrame(rows).sort_values("rmse_rel_naive").to_csv(
                tab_dir / "T7_vs_naive_benchmark.csv", index=False
            )
            written["T7"] = (
                "every model vs the expanding-mean benchmark, with separate "
                "verdicts for mean loss (wild bootstrap) and the typical "
                "quarter (sign test)"
            )
        else:
            skipped["T7"] = "no model overlaps the naive benchmark enough"
    else:
        skipped["T7"] = f"benchmark '{naive_ref}' not among the run's models"

    # T12 -- the fine-tuning budget sweep, BOTH routes. This was previously
    # `results/m4_finetune_budget.csv`, written by an ad-hoc script that no
    # longer exists and that covered only the LoRA arms, so the study's
    # "within +-0.2% of zero-shot" was computed from half the sweep
    #. Generated here from the predictions so the
    # table cannot be partial.
    zero_shot = ev.get("finetune_zero_shot_model", "chronos_ragged_edge")
    rows = []
    if zero_shot in wide_all.columns:
        ft_models = [
            m
            for m in wide_all.columns
            if ("lora_ft" in m or "full_ft" in m) and m != zero_shot
        ]
        for model in sorted(ft_models):
            pair = wide_all[[model, zero_shot]].dropna()
            if len(pair) < 50:
                continue
            digits = "".join(c for c in model.split("_ft")[-1] if c.isdigit())
            q = dm_test_by_quarter(pair[model], pair[zero_shot])
            rows.append(
                {
                    "model": model,
                    "mode": "lora" if "lora_ft" in model else "full",
                    "finetune_steps": int(digits) if digits else 0,
                    "n_pairs": q["n_pairs"],
                    "n_quarters": q["n_quarters"],
                    "rmse": q["rmse_1"],
                    "rmse_zero_shot": q["rmse_2"],
                    "rmse_rel_zero_shot": q["rmse_1"] / q["rmse_2"],
                    "deviation_pct": 100 * (q["rmse_1"] / q["rmse_2"] - 1),
                    "p_t_quarterly": q["p_t"],
                    "p_sign": q["p_sign"],
                }
            )
    if rows:
        t12 = pd.DataFrame(rows).sort_values(["mode", "finetune_steps"])
        # The zero-shot row anchors the sweep at exactly zero deviation.
        anchor_row = pd.DataFrame(
            [
                {
                    "model": zero_shot,
                    "mode": "zero-shot",
                    "finetune_steps": 0,
                    "n_pairs": int(wide_all[zero_shot].notna().sum()),
                    "n_quarters": int(
                        wide_all[[zero_shot]]
                        .dropna()
                        .index.get_level_values("target_quarter")
                        .nunique()
                    ),
                    "rmse": float(np.sqrt((wide_all[zero_shot].dropna() ** 2).mean())),
                    "rmse_zero_shot": float(
                        np.sqrt((wide_all[zero_shot].dropna() ** 2).mean())
                    ),
                    "rmse_rel_zero_shot": 1.0,
                    "deviation_pct": 0.0,
                    "p_t_quarterly": float("nan"),
                    "p_sign": float("nan"),
                }
            ]
        )
        t12 = pd.concat([anchor_row, t12], ignore_index=True)
        t12.to_csv(tab_dir / "T12_finetune_budget.csv", index=False)
        worst = float(t12["deviation_pct"].abs().max())
        written["T12"] = (
            f"fine-tuning sweep, both routes: worst deviation {worst:.2f}% "
            f"from zero-shot across {len(t12) - 1} configurations"
        )
    else:
        skipped["T12"] = "no fine-tuned arms found in the predictions"

    # T9 -- the single-factor experiment across economies, under the
    # quarterly inference of §66. Regions are declared with their own results
    # and target paths because ALFRED's non-US vintage coverage forces a
    # different sample per economy (§64, §65).
    regions = cfg.get(
        "cross_economy",
        {
            "US": {"results": str(out_dir), "target": None},
            "euro area": {
                "results": "results_intl_sf",
                "target": "data/processed_intl/target_second_release.csv",
            },
            "France": {
                "results": "results_fr",
                "target": "data/processed_fr/target_second_release.csv",
            },
        },
    )
    rows = []
    for region, spec in regions.items():
        rdir = Path(spec["results"])
        if not (rdir / "preds").exists():
            continue
        try:
            rp = M.load_predictions(rdir)
            rp = rp[~rp["variant"].astype(str).str.contains("balanced")]
            ry = target if spec.get("target") is None else M.load_target(spec["target"])
            rprep = M.prepare(rp, ry)
        except Exception as exc:  # noqa: BLE001 - a missing region is not fatal
            logger.warning("T9: could not load %s (%s)", region, exc)
            continue
        rans = rprep[rprep["yhat"].notna()]
        rw = rans.pivot_table(
            index=["nowcast_date", "target_quarter"], columns="model", values="error"
        )
        rq = rw.index.get_level_values("target_quarter")
        is_covid = (rq >= pd.Timestamp("2020-01-01")) & (
            rq <= pd.Timestamp("2021-06-30")
        )
        for sample, sub in (("full", rw), ("ex_covid", rw[~is_covid])):
            for m1, m2 in (
                ["chronos_completed_dfm", "dfm_robust"],
                ["dfm_robust", naive_ref],
                ["chronos_completed_dfm", naive_ref],
            ):
                if m1 not in sub.columns or m2 not in sub.columns:
                    continue
                pair = sub[[m1, m2]].dropna()
                if len(pair) < 50:
                    continue
                q = dm_test_by_quarter(pair[m1], pair[m2])
                rows.append(
                    {
                        "region": region,
                        "sample": sample,
                        "model": m1,
                        "benchmark": m2,
                        "n_pairs": q["n_pairs"],
                        "n_quarters": q["n_quarters"],
                        "rmse_model": q["rmse_1"],
                        "rmse_benchmark": q["rmse_2"],
                        "rmse_ratio": q["rmse_1"] / q["rmse_2"],
                        "quarters_model_ahead": q["n_quarters_model_ahead"],
                        "p_t": q["p_t"],
                        "p_sign": q["p_sign"],
                        "p_wilcoxon": q["p_wilcoxon"],
                    }
                )
    if rows:
        t9 = pd.DataFrame(rows)
        t9.to_csv(tab_dir / "T9_cross_economy.csv", index=False)
        written["T9"] = (
            f"single-factor experiment in {t9['region'].nunique()} economies "
            f"under quarterly inference (§64-§66)"
        )
    else:
        skipped["T9"] = "no cross-economy run available"

    # T8 -- how much of this study's inference is the HAC bandwidth. This is
    # a methodological result, not a robustness footnote: at the bandwidth
    # this project used until §66 most headline comparisons are significant,
    # and at the bandwidth the overlap actually implies, none are.
    t8_pairs = ev.get(
        "inference_audit_pairs",
        [
            ["chronos_completed_dfm", "dfm_robust"],
            ["dfm", naive_ref],
            ["dfm_robust", naive_ref],
            ["chronos_completed_dfm", naive_ref],
            ["chronos_ragged_edge", naive_ref],
            ["midas", naive_ref],
            ["xgboost", naive_ref],
            # The other two arms the study reports as behind the benchmark in
            # most quarters: a claim about them needs them in the audit.
            ["random_forest", naive_ref],
            ["lstm", naive_ref],
            ["bridge", "dfm_robust"],
            ["chronos_aggregate_q", naive_ref],
        ],
    )
    rows = []
    for m1, m2 in t8_pairs:
        if m1 not in wide_all.columns or m2 not in wide_all.columns:
            continue
        pair = wide_all[[m1, m2]].dropna()
        if len(pair) < 50:
            continue
        sens = hac_sensitivity(pair[m1], pair[m2])
        q = dm_test_by_quarter(pair[m1], pair[m2])
        row = {
            "model": m1,
            "benchmark": m2,
            "n_pairs": len(pair),
            "n_quarters": q["n_quarters"],
            "rmse_model": q["rmse_1"],
            "rmse_benchmark": q["rmse_2"],
        }
        for _, r in sens.iterrows():
            row[f"p_hac{int(r['hac_lags'])}"] = r["p_value"]
        row["p_t_quarterly"] = q["p_t"]
        row["p_sign"] = q["p_sign"]
        row["p_wilcoxon"] = q["p_wilcoxon"]
        rows.append(row)
    if rows:
        t8 = pd.DataFrame(rows)
        t8.to_csv(tab_dir / "T8_inference_audit.csv", index=False)
        n_hac4 = int((t8["p_hac4"] < 0.05).sum())
        n_q = int((t8["p_t_quarterly"] < 0.05).sum())
        written["T8"] = (
            f"inference audit: {n_hac4}/{len(t8)} comparisons significant at "
            f"HAC(4), {n_q}/{len(t8)} under quarterly inference (§66)"
        )
    else:
        skipped["T8"] = "no comparison pair had enough overlap"

    # T13 -- the headline comparisons under EVERY inference procedure a
    # referee would ask about, not only the three of §66. The cluster-robust
    # family is the obvious answer to "the loss differential is clustered by
    # target quarter", so it is measured; §85 shows CR1 coincides exactly
    # with the quarter-aggregated t-test on a balanced grid, so the two
    # remedies are one procedure, and neither can reject on real data.
    from scipy import stats
    from scipy import stats as _st

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

    boot = int(ev.get("bootstrap_draws", 1999))
    rows = []
    for m1, m2 in t8_pairs:
        if m1 not in wide_all.columns or m2 not in wide_all.columns:
            continue
        pair = wide_all[[m1, m2]].dropna()
        if len(pair) < 50:
            continue
        d = (pair[m1] ** 2 - pair[m2] ** 2).to_numpy(float)
        g = pd.factorize(pair.index.get_level_values("target_quarter"))[0]
        dq = np.bincount(g, weights=d) / np.bincount(g)
        q = dm_test_by_quarter(pair[m1], pair[m2])
        weekly = dm_test(pair[m1], pair[m2], hac_lags=ev.get("dm_hac_lags", 4))
        try:
            p_wilcoxon = float(_st.wilcoxon(dq)[1])
        except ValueError:
            p_wilcoxon = float("nan")
        rows.append(
            {
                "model": m1,
                "benchmark": m2,
                "n_pairs": len(pair),
                "n_clusters": int(g.max()) + 1,
                "rmse_model": q["rmse_1"],
                "rmse_benchmark": q["rmse_2"],
                "p_weekly_hac": weekly[1],
                "p_cluster_cr1": cluster_robust(d, g)["p"],
                "p_cluster_cr2": cluster_robust_cr2(d, g)["p"],
                "p_cluster_cr3": cluster_jackknife(d, g)["p"],
                "p_wild_bootstrap": wild_cluster_bootstrap(
                    d, g, n_boot=boot, seed=cfg.get("seed", 42)
                )["p"],
                "p_randomization_mean": randomization_mean(
                    d, g, n_draws=boot, seed=cfg.get("seed", 42)
                )["p"],
                "p_quarterly_t": q["p_t"],
                "p_wilcoxon": p_wilcoxon,
                "p_sign": q["p_sign"],
                # Valid under weak temporal dependence between clusters,
                # which the diagnostics say we need (§96). Liberal at this G
                # by the size study, so their non-rejections are the
                # informative direction.
                "p_hac_quarterly": hac_quarterly(d, g)["p"],
                "p_moving_block": moving_block_bootstrap(
                    d, g, n_boot=boot, seed=cfg.get("seed", 42)
                )["p"],
                # Correctly sized under ALL FOUR synthetic processes,
                # serial dependence included, unlike the nine above (§99).
                "p_ibragimov_muller": ibragimov_muller(d, g, q=4)["p"],
                "p_hac_fixed_b": hac_quarterly_fixed_b(d, g, b=0.25)["p"],
            }
        )
    if rows:
        t13 = pd.DataFrame(rows)
        t13.to_csv(tab_dir / "T13_procedure_audit.csv", index=False)
        liberal = ["p_hac_quarterly", "p_moving_block"]
        sized = ["p_ibragimov_muller", "p_hac_fixed_b"]
        n_liberal = int((t13[liberal] < 0.05).any(axis=1).sum())
        n_sized = int((t13[sized] < 0.05).any(axis=1).sum())
        written["T13"] = (
            f"the headline comparisons under 13 inference procedures; "
            f"{n_liberal}/{len(t13)} significant under a procedure that allows "
            f"temporal dependence but is liberal at this G, and "
            f"{n_sized}/{len(t13)} under one that is correctly sized under "
            f"every synthetic process (§96, §99)"
        )
    else:
        skipped["T13"] = "no comparison pair had enough overlap"

    # T16 -- effective sample size. The study's two failure mechanisms have
    # an intuitive common expression: the nominal sample is 1,400 forecast
    # pairs, the design supplies 82 target-quarter clusters, and after
    # serial dependence and tail leverage the information content is far
    # smaller than either.
    rows = []
    for m1, m2 in t8_pairs:
        if m1 not in wide_all.columns or m2 not in wide_all.columns:
            continue
        pair = wide_all[[m1, m2]].dropna().sort_index(level="target_quarter")
        if len(pair) < 50:
            continue
        d = (pair[m1] ** 2 - pair[m2] ** 2).to_numpy(float)
        g = pd.factorize(pair.index.get_level_values("target_quarter"))[0]
        dq = np.bincount(g, weights=d) / np.bincount(g)
        n_g = len(dq)
        x = dq - dq.mean()
        ss_x = float((x**2).sum())
        if ss_x <= 0:
            continue
        # (1) Serial dependence: G divided by the Bartlett-weighted variance
        # inflation factor, the standard effective-sample-size correction for
        # an autocorrelated mean.
        lags = int(np.floor(4 * (n_g / 100.0) ** (2.0 / 9.0)))
        inflation = 1.0
        for lag in range(1, max(lags, 1) + 1):
            rho = float((x[lag:] * x[:-lag]).sum() / ss_x)
            inflation += 2.0 * (1.0 - lag / (lags + 1.0)) * rho
        ess_serial = n_g / max(inflation, 1e-9)
        # (2) Tail leverage: the Satterthwaite effective degrees of freedom
        # for the mean of heteroskedastic terms, each quarter's variance
        # estimated by its own squared deviation.
        nu = ss_x**2 / float((x**4).sum())
        # The DIAGNOSTIC reading of nu (a reference t at nu degrees of
        # freedom) is reported for transparency but is not what any verdict
        # rests on -- see the study's "What nu is not". The conventional
        # critical value is the one the exact range argument uses.
        t_crit = float(stats.t.ppf(0.975, max(nu, 1.001)))
        t_crit_conventional = float(stats.t.ppf(0.975, n_g - 1))
        # (3) The largest |t| the sign-randomization design can produce. The
        # sum of squares is invariant to sign flips, so the statistic is
        # monotone in |mean| and the maximum is attained at s = sign(dq):
        # a closed form, not a search.
        m_max = float(np.abs(dq).mean())
        var_max = (float((dq**2).sum()) - n_g * m_max**2) / (n_g * (n_g - 1))
        t_max = m_max / np.sqrt(var_max) if var_max > 0 else np.inf
        rows.append(
            {
                "model": m1,
                "benchmark": m2,
                "n_pairs": len(d),
                "n_clusters": n_g,
                "rho1": float((x[1:] * x[:-1]).sum() / ss_x),
                # Skewness of the quarterly differential. The size study's
                # equal-mean processes are skewed because these are, and the
                # paper quotes the range, so it is measured here rather than
                # asserted.
                "skew_quarterly": float(stats.skew(dq)),
                "variance_inflation": float(inflation),
                "ess_serial": float(ess_serial),
                "satterthwaite_df": float(nu),
                "t_critical_at_df": t_crit,
                "t_critical_conventional": t_crit_conventional,
                "t_max_randomization": float(t_max),
                "can_ever_reject": bool(t_max > t_crit_conventional),
            }
        )
    if rows:
        t16 = pd.DataFrame(rows)
        t16.to_csv(tab_dir / "T16_effective_sample.csv", index=False)
        written["T16"] = (
            f"effective sample size: {t16['n_pairs'].max():,} pairs, "
            f"{t16['n_clusters'].max()} clusters, "
            f"{t16['ess_serial'].min():.0f}-{t16['ess_serial'].max():.0f} after "
            f"serial dependence, {t16['satterthwaite_df'].min():.1f}-"
            f"{t16['satterthwaite_df'].max():.1f} effective df after tail "
            f"leverage; {int((~t16['can_ever_reject']).sum())} of "
            f"{len(t16)} comparisons have no attainable rejection region "
            f"under the sign-randomization design at all (§98, §99)"
        )
    else:
        skipped["T16"] = "no comparison pair had enough overlap"

    # T17 -- is the recommendation licensed HERE, and does it depend on the
    # tuning? The size study (§100) shows the two recommended procedures hold
    # their level under short-range dependence and fail under long-range
    # dependence, so the recommendation is conditional on which regime the
    # data is in. This table answers both questions from the data: the
    # autocorrelation profile says the regime, and the sweep over the
    # procedures' tuning parameters says whether any verdict rests on it.
    rows = []
    for m1, m2 in t8_pairs:
        if m1 not in wide_all.columns or m2 not in wide_all.columns:
            continue
        pair = wide_all[[m1, m2]].dropna().sort_index(level="target_quarter")
        if len(pair) < 50:
            continue
        d = (pair[m1] ** 2 - pair[m2] ** 2).to_numpy(float)
        g = pd.factorize(pair.index.get_level_values("target_quarter"))[0]
        dq = np.bincount(g, weights=d) / np.bincount(g)
        x = dq - dq.mean()
        ss = float((x * x).sum())
        if ss <= 0:
            continue
        rho = [float((x[k:] * x[:-k]).sum() / ss) for k in range(1, 9)]
        row = {
            "model": m1,
            "benchmark": m2,
            "n_clusters": len(dq),
            **{f"rho{k}": rho[k - 1] for k in range(1, 9)},
            # Short-range dependence shows up as a first lag that carries
            # nearly all of the absolute autocorrelation and long lags that
            # do not accumulate. Long-range dependence is the opposite, and
            # is the case in which the recommended procedures fail.
            "rho1_share_of_abs": abs(rho[0]) / sum(abs(v) for v in rho),
            "sum_rho_lags5_8": float(sum(rho[4:])),
        }
        for q in (2, 3, 4, 5, 6, 8):
            row[f"p_im_q{q}"] = ibragimov_muller(d, g, q=q)["p"]
        for b in (0.1, 0.25, 0.5):
            key = f"p_fixed_b{str(b).replace('.', '')}"
            row[key] = hac_quarterly_fixed_b(d, g, b=b)["p"]
        tuned = [v for k, v in row.items() if k.startswith(("p_im_", "p_fixed_b"))]
        row["p_min_over_tuning"] = min(tuned)
        row["p_max_over_tuning"] = max(tuned)
        row["verdict_stable"] = bool(max(tuned) < 0.05 or min(tuned) >= 0.05)
        rows.append(row)
    if rows:
        t17 = pd.DataFrame(rows)
        t17.to_csv(tab_dir / "T17_dependence_regime.csv", index=False)
        written["T17"] = (
            f"dependence regime and tuning sensitivity: rho1 carries "
            f"{100 * t17['rho1_share_of_abs'].min():.0f}-"
            f"{100 * t17['rho1_share_of_abs'].max():.0f}% of the absolute "
            f"autocorrelation and lags 5-8 sum to "
            f"{t17['sum_rho_lags5_8'].min():+.2f} to "
            f"{t17['sum_rho_lags5_8'].max():+.2f} (short-range); the verdict "
            f"is unchanged across all nine tunings in "
            f"{int(t17['verdict_stable'].sum())}/{len(t17)} comparisons (§100)"
        )
    else:
        skipped["T17"] = "no comparison pair had enough overlap"

    # T15 -- multiplicity. Eleven headline comparisons, twenty-one
    # model-versus-benchmark comparisons, two samples and four block lengths
    # generate a lot of tests, and a result at p = 0.046 among twenty-one is
    # not the same evidence as the same p-value on its own. Two hypotheses
    # are declared PRIMARY -- the architectural contrast and the question the
    # study was built for -- and everything else is a family under Holm and
    # Benjamini-Hochberg control.
    primary = ev.get(
        "primary_hypotheses",
        [
            ["dfm_robust", "bridge"],
            ["chronos_completed_dfm", "dfm_robust"],
        ],
    )
    from src.eval.cluster_inference import wild_cluster_bootstrap as _wcb

    def _dep_robust_p(m1: str, m2: str) -> tuple[float, int] | None:
        """p-value from the moving-block bootstrap on cluster means.

        Used for the primary family because it is the only procedure here
        that is valid under temporal dependence between clusters, which the
        data exhibits (§96). It is still oversized at G = 82 under the
        measured autocorrelation, so it is liberal rather than exact.
        """
        if m1 not in wide_all.columns or m2 not in wide_all.columns:
            return None
        pair = wide_all[[m1, m2]].dropna().sort_index(level="target_quarter")
        if len(pair) < 50:
            return None
        d = (pair[m1] ** 2 - pair[m2] ** 2).to_numpy(float)
        g = pd.factorize(pair.index.get_level_values("target_quarter"))[0]
        return (
            moving_block_bootstrap(
                d,
                g,
                n_boot=int(ev.get("bootstrap_draws", 1999)),
                seed=cfg.get("seed", 42),
            )["p"],
            int(g.max()) + 1,
        )

    def _wild_p(m1: str, m2: str) -> tuple[float, int] | None:
        if m1 not in wide_all.columns or m2 not in wide_all.columns:
            return None
        pair = wide_all[[m1, m2]].dropna()
        if len(pair) < 50:
            return None
        d = (pair[m1] ** 2 - pair[m2] ** 2).to_numpy(float)
        g = pd.factorize(pair.index.get_level_values("target_quarter"))[0]
        return (
            _wcb(
                d,
                g,
                n_boot=int(ev.get("bootstrap_draws", 1999)),
                seed=cfg.get("seed", 42),
            )["p"],
            int(g.max()) + 1,
        )

    def _im_p(m1: str, m2: str) -> tuple[float, int] | None:
        """p-value from the Ibragimov-Muller t-test on contiguous groups.

        The primary family's verdict, because it is the procedure whose
        assumptions the data satisfies: correctly sized under every process
        in the size study, serial dependence included (§99, §100).
        """
        if m1 not in wide_all.columns or m2 not in wide_all.columns:
            return None
        pair = wide_all[[m1, m2]].dropna().sort_index(level="target_quarter")
        if len(pair) < 50:
            return None
        d = (pair[m1] ** 2 - pair[m2] ** 2).to_numpy(float)
        g = pd.factorize(pair.index.get_level_values("target_quarter"))[0]
        return ibragimov_muller(d, g, q=4)["p"], int(g.max()) + 1

    rows = []
    # The primary family is reported under BOTH the procedure whose
    # assumptions the data satisfies and the one that assumes independent
    # target quarters, each labelled with that assumption. Reporting only one
    # invited exactly the confusion §100 records: the prose quoted the
    # cluster-independent p-value while the table showed the other.
    for m1, m2 in primary:
        for fam, proc, getter, dep in (
            ("primary", "Ibragimov-Muller", _im_p, "allows dependence"),
            (
                "primary_cluster_indep",
                "wild cluster bootstrap",
                _wild_p,
                "assumes independence",
            ),
        ):
            got = getter(m1, m2)
            if got is None:
                continue
            rows.append(
                {
                    "family": fam,
                    "model": m1,
                    "benchmark": m2,
                    "n_clusters": got[1],
                    "p_raw": got[0],
                    "estimand": "mean loss",
                    "procedure": proc,
                    "dependence": dep,
                }
            )
    # Two families, one per estimand: mean loss by the wild bootstrap and the
    # typical quarter by the sign test. Adjusting within estimand rather than
    # pooling them is the right unit, since a claim about mean loss and a
    # claim about the median quarter are different hypotheses.
    for model in sorted(wide_all.columns):
        if model == naive_ref or [model, naive_ref] in primary:
            continue
        got = _wild_p(model, naive_ref)
        if got is None:
            continue
        rows.append(
            {
                "family": "mean_loss_vs_benchmark",
                "model": model,
                "benchmark": naive_ref,
                "n_clusters": got[1],
                "p_raw": got[0],
                "estimand": "mean loss",
                "procedure": "wild cluster bootstrap",
                "dependence": "assumes independence",
            }
        )
        pair = wide_all[[model, naive_ref]].dropna()
        rows.append(
            {
                "family": "typical_quarter_vs_benchmark",
                "model": model,
                "benchmark": naive_ref,
                "n_clusters": got[1],
                "p_raw": dm_test_by_quarter(pair[model], pair[naive_ref])["p_sign"],
                "estimand": "typical quarter",
                "procedure": "exact sign test",
                "dependence": "assumes independent signs",
            }
        )
    if rows:
        t15 = pd.DataFrame(rows)
        for fam, grp in t15.groupby("family"):
            idx = grp.sort_values("p_raw").index
            raw = t15.loc[idx, "p_raw"].to_numpy(float)
            m = len(raw)
            # Holm: step-down, p_(i) * (m - i + 1), enforced monotone.
            holm = np.minimum.accumulate(
                np.minimum(raw * (m - np.arange(m)), 1.0)[::-1]
            )[::-1]
            holm = np.maximum.accumulate(np.minimum(raw * (m - np.arange(m)), 1.0))
            # Benjamini-Hochberg: p_(i) * m / i, enforced monotone from the top.
            bh = np.minimum.accumulate((raw * m / (np.arange(m) + 1))[::-1])[::-1]
            t15.loc[idx, "p_holm"] = np.minimum(holm, 1.0)
            t15.loc[idx, "p_bh"] = np.minimum(bh, 1.0)
            t15.loc[idx, "family_size"] = m
        t15 = t15.sort_values(["family", "p_raw"])

        # The status column exists so that "significant conditional on
        # independent target quarters" can never be read as the study's
        # verdict. A row whose procedure assumes independence is capped at
        # "conditional", however small its p-value, because §96's runs test
        # rejects that assumption for the comparison it matters for.
        def _status(r) -> str:
            if not np.isfinite(r["p_holm"]):
                return "not computed"
            if r["p_raw"] >= 0.05:
                return "not significant"
            conditional = r["dependence"] != "allows dependence"
            if r["p_holm"] < 0.05:
                level = "significant after Holm"
            elif r["p_bh"] < 0.05:
                level = "significant after BH only"
            else:
                level = "significant unadjusted only"
            return f"{level}; conditional on independence" if conditional else level

        t15["status"] = t15.apply(_status, axis=1)
        t15.to_csv(tab_dir / "T15_multiplicity.csv", index=False)
        n_raw = int((t15["p_raw"] < 0.05).sum())
        n_holm = int((t15["p_holm"] < 0.05).sum())
        n_bh = int((t15["p_bh"] < 0.05).sum())
        written["T15"] = (
            f"multiplicity: {n_raw} significant unadjusted, {n_holm} under "
            f"Holm, {n_bh} under Benjamini-Hochberg (§91)"
        )
    else:
        skipped["T15"] = "no comparison pair had enough overlap"

    # T14 -- robustness of every surviving result to BETWEEN-cluster
    # dependence. The quarterly loss differential carries first-order
    # autocorrelation of +0.32 to +0.48 here, so independent per-quarter sign
    # flips are not licensed; flipping contiguous blocks of quarters together
    # preserves sign dependence up to the block length (§86). A result that
    # holds at block 4 tolerates a year of serial dependence; one that holds
    # at block 8 tolerates two.
    from src.eval.cluster_inference import _block_weights

    blocks = ev.get("randomization_blocks", [1, 2, 4, 8])
    rows = []
    for m1, m2 in t8_pairs:
        if m1 not in wide_all.columns or m2 not in wide_all.columns:
            continue
        pair = wide_all[[m1, m2]].dropna().sort_index(level="target_quarter")
        if len(pair) < 50:
            continue
        d = (pair[m1] ** 2 - pair[m2] ** 2).to_numpy(float)
        g = pd.factorize(pair.index.get_level_values("target_quarter"))[0]
        dq = np.bincount(g, weights=d) / np.bincount(g)
        n_g = len(dq)
        # Serial dependence in the magnitudes and in the signs.
        centred = dq - dq.mean()
        denom = float((centred**2).sum())
        rho1 = float((centred[1:] * centred[:-1]).sum() / denom) if denom > 0 else 0.0
        signs = (dq < 0).astype(int)
        n_runs = 1 + int((np.diff(signs) != 0).sum())
        n1, n0 = int(signs.sum()), n_g - int(signs.sum())
        if n1 and n0:
            mu = 2 * n1 * n0 / n_g + 1
            sd = np.sqrt(2 * n1 * n0 * (2 * n1 * n0 - n_g) / (n_g**2 * (n_g - 1)))
            p_runs = (
                float(2 * (1 - stats.norm.cdf(abs(n_runs - mu) / sd)))
                if sd > 0
                else float("nan")
            )
        else:
            p_runs = float("nan")
        row = {
            "model": m1,
            "benchmark": m2,
            "n_clusters": n_g,
            "rho1_quarterly": rho1,
            "p_runs_on_signs": p_runs,
        }
        rng_seed = cfg.get("seed", 42)
        draws = int(ev.get("bootstrap_draws", 1999))
        for b in blocks:
            row[f"p_wild_block{b}"] = wild_cluster_bootstrap(
                d, g, n_boot=draws, seed=rng_seed, block=int(b)
            )["p"]
            # Block randomization of the sign statistic: the exact sign test
            # assumes independent signs, so its block analogue is computed by
            # resampling rather than from the binomial.
            rng = np.random.default_rng(rng_seed)
            weights = _block_weights(rng, draws, n_g, int(b))
            observed = abs(int((dq < 0).sum()) - n_g / 2)
            flipped = (dq[None, :] * weights) < 0
            stat = np.abs(flipped.sum(axis=1) - n_g / 2)
            row[f"p_sign_block{b}"] = float(
                (1 + int((stat >= observed).sum())) / (1 + draws)
            )
        rows.append(row)
    if rows:
        t14 = pd.DataFrame(rows)
        t14.to_csv(tab_dir / "T14_block_robustness.csv", index=False)
        cols = [
            c for c in t14.columns if c.startswith(("p_wild_block", "p_sign_block"))
        ]
        n_any = int((t14[cols] < 0.05).any(axis=1).sum())
        written["T14"] = (
            f"block-randomization robustness to between-cluster dependence; "
            f"{n_any}/{len(t14)} comparisons significant under at least one "
            f"block length (§86)"
        )
    else:
        skipped["T14"] = "no comparison pair had enough overlap"

    # F5 -- how much of the error the publication cycle actually recovers,
    # pooled across models. F2 shows one line per model; F5 answers the single
    # question a policymaker asks: what does waiting another week buy?
    if not dec.empty and dec["cycle_week"].notna().any():
        # A pooled profile across all cycle weeks is not interpretable: the
        # late weeks are covered by a handful of quarters, and models that
        # abstain early enter the pool later, so a raw pooled line mixes the
        # information effect with changes in sample composition. Restrict to
        # cycle weeks with FULL quarter coverage, then to (model, quarter)
        # cells observed at every one of them.
        nq_by_week = dec.groupby("cycle_week")["target_quarter"].nunique()
        weeks = sorted(nq_by_week[nq_by_week == nq_by_week.max()].index, reverse=True)
        bal = dec[dec["cycle_week"].isin(weeks)]
        cells = bal.groupby(["model", "target_quarter"])["cycle_week"].nunique()
        keep = set(cells[cells == len(weeks)].index)
        bal = bal[[(m, q) in keep for m, q in zip(bal["model"], bal["target_quarter"])]]
    else:
        bal, weeks = pd.DataFrame(), []

    if not bal.empty and len(weeks) > 2:
        first_week, last_week = weeks[0], weeks[-1]
        by_mw = bal.pivot_table(
            index="model",
            columns="cycle_week",
            values="error",
            aggfunc=lambda e: float(np.sqrt((np.asarray(e, float) ** 2).mean())),
        )
        # Order weeks from earliest to latest in the cycle so that idxmin
        # resolves ties towards the EARLIEST week. Ties are real: a model fed
        # only quarterly GDP (the aggregate_q arms, the naive mean) cannot
        # change its nowcast until a new GDP release arrives, so its profile
        # is flat for stretches of the quarter.
        by_mw = by_mw[sorted(by_mw.columns, reverse=True)]
        rows = []
        for model, r in by_mw.iterrows():
            best_week = int(r.idxmin())
            rows.append(
                {
                    "model": model,
                    "rmse_first_week": float(r[first_week]),
                    "rmse_best_week": float(r.min()),
                    "best_week": best_week,
                    "rmse_last_week": float(r[last_week]),
                    "gain_to_best_pct": 100 * (1 - r.min() / r[first_week]),
                    "gain_at_end_pct": 100 * (1 - r[last_week] / r[first_week]),
                }
            )
        f5 = pd.DataFrame(rows).sort_values("gain_to_best_pct", ascending=False)
        f5["first_week"] = first_week
        f5["last_week"] = last_week
        f5["n_quarters"] = bal["target_quarter"].nunique()
        f5.to_csv(tab_dir / "F5_within_quarter_cycle.csv", index=False)
        by_mw.to_csv(tab_dir / "F5_cycle_profile_balanced.csv")

        written["F5"] = (
            f"per-model cycle gain on a balanced panel "
            f"(weeks {first_week}..{last_week}, {bal['target_quarter'].nunique()}q)"
        )
    else:
        skipped["F5"] = "no cycle week has full quarter coverage"

    # DM table is appendix C material.
    # hac_lags omitted on purpose: dm_table derives it from the grid.
    dm = dm_table(common, benchmark="dfm", hac_lags=ev.get("dm_hac_lags"))
    dm.to_csv(tab_dir / "TC1_diebold_mariano_vs_dfm.csv", index=False)
    written["TC1"] = "pairwise Diebold-Mariano against the DFM"

    # F2 -- RMSE by week of the publication cycle (RQ2). CSV only: figures
    # are built by src/viz/figures.py against a validated design system
    #, so this stage no longer draws.
    profile = M.cycle_profile(preds, target)
    if not profile.empty:
        profile.to_csv(tab_dir / "F2_cycle_profile.csv", index=False)
        written["F2"] = "RMSE by week of the publication cycle"
    else:
        skipped["F2"] = "no common-sample profile available"

    # F3 -- superseded by T12_finetune_budget.csv, which covers BOTH
    # fine-tuning routes; the m4 artifact held only the LoRA arms (§72).
    m4_path = out_dir / "m4_finetune_budget.csv"
    if m4_path.exists():
        written["F3"] = "superseded by T12; see"
    else:
        skipped["F3"] = "fine-tuning budget sweep (M4) not run"

    # F1 -- the ragged-edge mask as a CSV. The figure itself is built by
    # src/viz/figures.py; the previous raster rendering here was replaced
    # after review.
    try:
        f1_series = alfred_client.panel_series_ids(data_cfg, "baseline")
        f1_t = pd.Timestamp(cfg.get("f1_date", "2015-05-12"))
        f1_q = f1_t.to_period("Q").start_time
        panel_f1 = info_set.information_set(
            f1_t,
            f1_series,
            target_id=data_cfg["target"]["fred_id"],
            target_quarter=f1_q,
        )
        window = panel_f1.loc[f1_t - pd.Timedelta(days=400) : f1_t]
        mask = window.notna().astype(int)
        last_obs = {c: window[c].last_valid_index() for c in window.columns}
        order = [
            c
            for c, _ in sorted(
                last_obs.items(),
                key=lambda kv: (kv[1] is None, kv[1] or pd.Timestamp.min),
            )
        ]
        mask[order].to_csv(tab_dir / "F1_ragged_edge_mask.csv")
        written["F1"] = "ragged edge of the real information set (data-derived)"
    except Exception as exc:  # noqa: BLE001 - must not fail the stage
        skipped["F1"] = f"could not build from data: {exc}"

    return {
        "tables_dir": str(tab_dir),
        "written": written,
        "skipped": skipped,
    }


def package_versions() -> dict:
    """Installed versions of the packages results depend on (R4)."""
    import importlib.metadata as md

    names = [
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "requests",
        "pyyaml",
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return versions


def write_manifest(
    out: Path,
    cfg: dict,
    run_id: str,
    stage_result: dict,
    duration_s: float,
    started_at: str,
) -> None:
    manifest = {
        "run_id": run_id,
        "stage": out.name.split("_", 1)[0],
        "config_hash": config_hash(cfg),
        "config": cfg,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "seed": cfg.get("seed"),
        "started_at": started_at,
        "duration_seconds": duration_s,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "result": stage_result,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage", required=True, choices=["download", "all", "evaluate", "tables"]
    )
    parser.add_argument(
        "--ablation", default=None, choices=[None, "ragged_edge", "balanced_panel"]
    )
    parser.add_argument(
        "--families",
        nargs="*",
        default=None,
        choices=sorted(set(MODEL_FAMILIES) | {"tsfm_ft"}),
        help="restrict the backtest to these model families",
    )
    parser.add_argument(
        "--panel",
        default="all",
        choices=list(alfred_client.PANELS),
        help="which panel tier to download (baseline carries headline results)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="root log level for this run",
    )
    args = parser.parse_args()

    # Configure logging BEFORE anything else. Without this every
    # logger.warning in the codebase is silently discarded, which is how a
    # full-fine-tuning run produced 1,403 NaN predictions with an empty log
    # and no indication of why.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    set_global_seed(cfg)
    run_id = f"{args.stage}_{config_hash(cfg)}"
    out = Path(cfg.get("output_dir", "results")) / run_id

    start = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    if args.stage == "download":
        result = run_stage_download(cfg, panel=args.panel)
    elif args.stage == "all":
        # `--families` and `--ablation` were parsed and then dropped here, so
        # both flags were accepted and silently did nothing: a run restricted
        # to one family quietly refitted every model in the config.
        result = run_stage_all(
            cfg,
            run_id,
            panel=args.panel if args.panel != "all" else "baseline",
            families=tuple(args.families) if args.families else None,
            ablation=args.ablation,
        )
    elif args.stage == "evaluate":
        result = run_stage_evaluate(cfg)
    elif args.stage == "tables":
        result = run_stage_tables(cfg)
    else:
        raise ValueError(f"unknown stage: {args.stage}")

    # Created only once the stage has actually succeeded, so a failed run does
    # not leave an empty results/<run_id>/ behind with no manifest in it.
    out.mkdir(parents=True, exist_ok=True)
    write_manifest(out, cfg, run_id, result, time.time() - start, started_at)
    return 0


if __name__ == "__main__":
    sys.exit(main())
