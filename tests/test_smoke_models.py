"""Smoke tests: every model fits and predicts on a tiny panel (M1-M4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _tiny_panel() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(60, 5)), columns=list("abcde"))
    y = pd.Series(rng.normal(size=60))
    return X, y


@pytest.mark.parametrize("model_name", ["bridge", "midas", "dfm"])
def test_econ_baselines_smoke(model_name):
    pytest.skip("M1")


@pytest.mark.parametrize("model_name", ["random_forest", "xgboost", "lstm"])
def test_ml_baselines_smoke(model_name):
    pytest.skip("M2")


@pytest.mark.parametrize("model_name", ["chronos", "lagllama", "timesfm"])
def test_tsfm_zero_shot_smoke(model_name):
    """Zero-shot wrappers load open weights and emit mean+quantiles."""
    pytest.skip("M3")
