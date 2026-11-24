"""Display names for models and synthetic processes, shared by tables and figures.

A reader who meets ``dfm_robust`` in a figure and ``Factor model (robust)`` in
the table beside it has to work out that they are the same thing. The mapping
therefore lives in one module that both the table builder and the figure
module import, and ``tests/test_labels.py`` fails if a model reaches the
results without a name here.

Names are the study's vocabulary, not the code's: ``chronos_ragged_edge`` is a
foundation model feeding a bridge equation, and that is what the reader needs
to see.
"""

from __future__ import annotations

MODEL_NAMES: dict[str, str] = {
    "dfm": "Factor model",
    "dfm_robust": "Factor model (robust)",
    "bridge": "Bridge equation",
    "midas": "MIDAS",
    "naive_mean": "Expanding mean",
    "naive_ar1": "AR(1)",
    "random_forest": "Random forest",
    "xgboost": "Gradient boosting",
    "lstm": "LSTM",
    "chronos_completed_dfm": "Factor model, edge completed (Chronos-Bolt)",
    "timesfm_completed_dfm": "Factor model, edge completed (TimesFM)",
    "lagllama_completed_dfm": "Factor model, edge completed (Lag-Llama)",
    "chronos2_completed_dfm": "Factor model, edge completed (Chronos-2)",
    "chronos2_cov_completed_dfm": "Factor model, edge completed (Chronos-2, covariates)",
    "chronos_ragged_edge": "Chronos-Bolt + bridge",
    "chronos_aggregate_q": "Chronos-Bolt, quarterly",
    "timesfm_ragged_edge": "TimesFM + bridge",
    "timesfm_aggregate_q": "TimesFM, quarterly",
    "lagllama_ragged_edge": "Lag-Llama + bridge",
    "lagllama_aggregate_q": "Lag-Llama, quarterly",
    # The fine-tuning sweep: two routes at three budgets each. The budget is
    # the number of update steps and belongs in the name, because the study's
    # null is a null about small updates rather than about fine-tuning.
    "chronos_lora_ft50": "Chronos, LoRA (50 steps)",
    "chronos_lora_ft200": "Chronos, LoRA (200 steps)",
    "chronos_lora_ft800": "Chronos, LoRA (800 steps)",
    "chronos_full_ft50": "Chronos, full fine-tune (50 steps)",
    "chronos_full_ft200": "Chronos, full fine-tune (200 steps)",
    "chronos_full_ft800": "Chronos, full fine-tune (800 steps)",
}

# Shorter forms, for marks that are direct-labelled inside a plot where the
# full name would collide with the data it sits beside.
MODEL_NAMES_SHORT: dict[str, str] = {
    "dfm": "Factor model",
    "dfm_robust": "Factor (robust)",
    "bridge": "Bridge",
    "midas": "MIDAS",
    "naive_mean": "Expanding mean",
    "naive_ar1": "AR(1)",
    "random_forest": "Random forest",
    "xgboost": "Boosting",
    "lstm": "LSTM",
    "chronos_completed_dfm": "Factor, edge compl. (Chronos-Bolt)",
    "timesfm_completed_dfm": "Factor, edge compl. (TimesFM)",
    "lagllama_completed_dfm": "Factor, edge compl. (Lag-Llama)",
    "chronos2_completed_dfm": "Factor, edge compl. (Chronos-2)",
    "chronos2_cov_completed_dfm": "Factor, edge compl. (Chronos-2, cov.)",
    "chronos_ragged_edge": "Chronos + bridge",
    "chronos_aggregate_q": "Chronos, quarterly",
    "timesfm_ragged_edge": "TimesFM + bridge",
    "timesfm_aggregate_q": "TimesFM, quarterly",
    "lagllama_ragged_edge": "Lag-Llama + bridge",
    "lagllama_aggregate_q": "Lag-Llama, quarterly",
    "chronos_lora_ft50": "LoRA 50",
    "chronos_lora_ft200": "LoRA 200",
    "chronos_lora_ft800": "LoRA 800",
    "chronos_full_ft50": "Full FT 50",
    "chronos_full_ft200": "Full FT 200",
    "chronos_full_ft800": "Full FT 800",
}

# The synthetic processes of the size study.
PROCESS_NAMES: dict[str, str] = {
    "overlap": "overlap",
    "serial": "serial",
    "heavy_tail": "heavy tail",
    "unbalanced": "unbalanced",
    "serial_strong": "persistent",
    "long_memory": "near unit root",
    "regime": "regime break",
    "few_clusters": "few clusters",
}


def model_name(key: str, *, short: bool = False) -> str:
    """Display name for a model key, falling back to the key itself.

    The fallback keeps a figure buildable when a new model appears; the test
    suite is what makes the omission visible.
    """
    table = MODEL_NAMES_SHORT if short else MODEL_NAMES
    return table.get(key, key)


# Just the completer, for tables whose column header already says that every
# row is the same factor model with its ragged edge filled differently.
COMPLETER_NAMES = {
    "chronos2_cov_completed_dfm": "Chronos-2, covariates",
    "chronos2_completed_dfm": "Chronos-2, rectangular context",
    "chronos_completed_dfm": "Chronos-Bolt",
    "timesfm_completed_dfm": "TimesFM",
    "lagllama_completed_dfm": "Lag-Llama",
}
