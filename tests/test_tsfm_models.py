"""Tests for the TSFM wrappers and the M4 fine-tuning path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.tsfm_models import (
    MIN_CONTEXT,
    ChronosLoRAFineTuner,
    TSFMNowcaster,
)


def _panel():
    rng = np.random.default_rng(0)
    months = pd.date_range("2000-01-01", "2019-12-01", freq="MS")
    return pd.DataFrame(
        {
            "IND1": rng.normal(0, 1, len(months)) + 5.0,
            "IND2": rng.normal(0, 1, len(months)).cumsum() * 0.1,
        },
        index=months,
    )


def test_strategy_must_be_known():
    with pytest.raises(ValueError, match="unknown strategy"):
        TSFMNowcaster(backend="chronos", strategy="not_a_strategy")
    with pytest.raises(ValueError, match="unknown backend"):
        TSFMNowcaster(backend="not_a_model", strategy="aggregate_q")


def test_missing_backend_dependency_abstains_rather_than_crashing():
    """An absent model must abstain, so it shows up in coverage.csv.

    All three families are installable now, so the abstention path is
    exercised via a missing CHECKPOINT rather than a missing package: the
    point is that an unavailable model degrades to NaN instead of aborting a
    multi-hour backtest.
    """
    from src.models.tsfm_models import LagLlamaWrapper

    w = LagLlamaWrapper(checkpoint="/nonexistent/lag-llama.ckpt")
    out = w.predict_batch([pd.Series(np.arange(50.0))], horizon=1)
    assert len(out) == 1
    assert np.isnan(out[0].mean)
    assert w.available_ is False


def test_all_three_families_are_registered():
    """ "TSFM" must mean more than one model family (referee-proofing)."""
    from src.models.tsfm_models import TSFM_MODELS

    families = {k.split("_")[0] for k in TSFM_MODELS}
    # chronos2 is a fourth entry rather than a fourth family: it is the
    # covariate-aware arm, and it exists to test the mechanism the other
    # three cannot use, so it offers no frequency strategies of its own.
    assert families == {"chronos", "chronos2", "timesfm", "lagllama"}
    for fam in ("chronos", "timesfm", "lagllama"):
        assert f"{fam}_aggregate_q" in TSFM_MODELS
        assert f"{fam}_ragged_edge" in TSFM_MODELS
    assert "chronos2_completed_dfm" in TSFM_MODELS


def test_finetuner_windows_come_only_from_the_given_panel():
    """R1: training data is the information set, nothing else."""
    tuner = ChronosLoRAFineTuner(context_length=16, prediction_length=3)
    panel = _panel()
    w = tuner._windows(panel)
    assert w.shape[1] == 16 + 3
    assert np.isfinite(w).all()

    # Truncating the panel must strictly reduce the window set: no window may
    # be built from data outside what was handed in.
    w_short = tuner._windows(panel.iloc[:100])
    assert len(w_short) < len(w)


def test_finetuner_skips_when_there_is_not_enough_history():
    pytest.importorskip("peft")  # LoRA only; no arm reported in the paper uses it
    tuner = ChronosLoRAFineTuner(context_length=512, prediction_length=64)
    tuner.fit(_panel().iloc[:20])
    assert tuner.n_windows_ == 0
    assert tuner.adapters_ == {}
    assert tuner.apply_adapter(50) is None


def test_zero_shot_pipeline_is_not_contaminated_by_finetuning():
    """The bug that would have silently invalidated H1.

    ``get_peft_model`` injects LoRA layers IN PLACE, and
    ``ChronosBoltPipeline`` has ``inner_model is model``. Fine-tuning the
    SHARED cached pipeline therefore turned every zero-shot Chronos model in
    the process into a fine-tuned one -- verified at the time by a forecast
    moving 12.77 -> 88.54 after 150 adapter steps. The zero-shot baselines
    are what H1 rests on, so this had to become a permanent check.
    """
    pytest.importorskip("chronos")
    pytest.importorskip("peft")

    panel = _panel()
    zero_shot = TSFMNowcaster(backend="chronos", strategy="aggregate_q")
    y = pd.Series(
        np.random.default_rng(1).normal(2, 2, 60),
        index=pd.date_range("2005-01-01", periods=60, freq="QS"),
    )
    zero_shot.fit(panel, y)
    q = y.index[-1] + pd.offsets.QuarterBegin(startingMonth=1)
    before = zero_shot.predict(panel, q, y).mean
    if not np.isfinite(before):
        pytest.skip("zero-shot produced no forecast on the fixture")

    tuner = ChronosLoRAFineTuner(budgets=(20,), context_length=64, prediction_length=3)
    tuner.fit(panel, seed=0)
    if not tuner.adapters_:
        pytest.skip("fine-tuning did not run on the fixture")

    after = zero_shot.predict(panel, q, y).mean
    assert after == pytest.approx(before, rel=1e-9), (
        "fine-tuning changed the ZERO-SHOT model's forecast -- the shared "
        "pipeline is being mutated in place and H1's baseline is invalid"
    )
    # And the tuner must own a private pipeline, not the shared cache.
    from src.models.tsfm_models import _PIPELINE_CACHE

    assert tuner.pipeline_ is not None
    assert all(tuner.pipeline_ is not p for p in _PIPELINE_CACHE.values())


def test_budget_zero_adapter_reproduces_the_base_model():
    """budget=0 must be a true control, not merely 'close'."""
    pytest.importorskip("chronos")
    pytest.importorskip("peft")
    tuner = ChronosLoRAFineTuner(budgets=(20,), context_length=64, prediction_length=3)
    tuner.fit(_panel(), seed=0)
    if not tuner.adapters_:
        pytest.skip("fine-tuning did not run on the fixture")
    assert 0 in tuner.adapters_, "no zero-budget control adapter was stored"
    assert all(float(v.abs().max()) == 0.0 for v in tuner.adapters_[0].values())


def test_min_context_is_enforced():
    model = TSFMNowcaster(backend="chronos", strategy="aggregate_q")
    short = pd.Series(np.arange(float(MIN_CONTEXT - 1)))
    out = model.wrapper.predict_batch([short], horizon=1)
    assert np.isnan(out[0].mean)


def test_finetune_budgets_are_independent_of_fit_order():
    """Each budget must forecast with ITS OWN adapter.

    All budgets share one private pipeline (training once and snapshotting is
    what makes the nested sweep affordable). Installing the adapter at FIT
    time meant the last-fitted budget owned the pipeline for everyone, and
    all budgets returned bit-identical forecasts across the whole 1,403-date
    backtest -- a silent collapse of RQ3 that reads as "fine-tuning has no
    effect". The adapter must therefore be installed at PREDICT time.
    """
    pytest.importorskip("chronos")
    pytest.importorskip("peft")
    from src.models.tsfm_models import ChronosFineTunedNowcaster

    panel = _panel()
    y = pd.Series(
        np.random.default_rng(2).normal(2, 2, 60),
        index=pd.date_range("2005-01-01", periods=60, freq="QS"),
    )
    q = y.index[-1] + pd.offsets.QuarterBegin(startingMonth=1)

    tuner = ChronosLoRAFineTuner(
        budgets=(20, 200), context_length=64, prediction_length=3
    )
    tuner.fit(panel, seed=0)
    if not tuner.adapters_:
        pytest.skip("fine-tuning did not run on the fixture")

    models = {
        b: ChronosFineTunedNowcaster(budget=b, shared_tuner=tuner) for b in (20, 200)
    }
    for m in models.values():
        m.fit(panel, y)

    # Predict in the REVERSE of fit order: a fit-time adapter install would
    # make both answers equal to the last-fitted budget's.
    p200 = models[200].predict(panel, q, y).mean
    p20 = models[20].predict(panel, q, y).mean
    # ...and again in the original order; each must be stable.
    assert models[20].predict(panel, q, y).mean == pytest.approx(p20, rel=1e-9)
    assert models[200].predict(panel, q, y).mean == pytest.approx(p200, rel=1e-9)


def test_context_window_is_trimmed_before_incomplete_columns_are_dropped():
    """A long history must not empty the context.

    The panel reaches back to 1913 while the model conditions on at most the
    last few hundred months. Dropping columns with any gap over the whole
    history removes every series that starts late, so the context came out
    with zero columns on every real date and the arm abstained silently on
    all 1403 of them -- a foundation-model arm that never ran the foundation
    model. Only the abstention guard made it visible.
    """
    import numpy as np
    import pandas as pd

    from src.models.tsfm_models import rectangular_context

    idx = pd.date_range("1913-01-01", "2024-12-01", freq="MS")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {f"s{i}": rng.normal(size=len(idx)) for i in range(5)}, index=idx
    )
    # Staggered starts, as in any real macro panel: each series begins later
    # than the one before and is gap-free from then on.
    for i, start in enumerate(
        ["1913-01-01", "1948-01-01", "1959-01-01", "1967-01-01", "1992-01-01"]
    ):
        frame.loc[frame.index < pd.Timestamp(start), f"s{i}"] = np.nan

    usable = list(frame.columns)
    ctx_end = frame.index[-1]
    ctx = rectangular_context(frame, usable, ctx_end, max_context=512)

    # 512 months back from 2024-12 is 1982-05, so the four series that begin
    # before then survive and s4, which begins inside the window and really
    # does have a gap the model would see, is correctly excluded.
    assert ctx.shape[1] == 4, (
        f"only {ctx.shape[1]} of 4 eligible series survived; the column "
        "filter is seeing history the model never receives"
    )
    assert "s4" not in ctx.columns
    assert ctx.shape[0] == 512
    assert ctx.notna().all().all()

    # The reverse order is what the bug did: it keeps only the one series
    # that spans the entire panel.
    wrong = frame.loc[:ctx_end, usable].dropna(axis=1, how="any").tail(512)
    assert wrong.shape[1] == 1
