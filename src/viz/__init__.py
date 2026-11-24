"""Figure styling and figure builders for the study.

Presentation layer only: nothing here computes a result. Every figure reads
an artifact produced under ``results/`` and draws it.
"""

from src.viz.style import (
    CATEGORICAL,
    INK,
    ORDINAL_BLUE,
    apply_style,
    emphasis_colors,
    end_labels,
    finish,
    label_end,
    legend_below,
)

__all__ = [
    "CATEGORICAL",
    "INK",
    "ORDINAL_BLUE",
    "apply_style",
    "emphasis_colors",
    "end_labels",
    "finish",
    "label_end",
    "legend_below",
]
