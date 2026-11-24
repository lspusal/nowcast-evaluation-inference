"""The figure style's colour claims, checked rather than asserted.

`src/viz/style.py` documents a validated palette. Two of its claims are
cheap to verify here and expensive to discover broken in a proof: that the
surface is the page's own white, and that no palette or ink token loses
contrast against it. The CVD separation gates are properties of the palette
colours alone and do not involve the surface, so they are unaffected by the
one substitution this file guards.
"""

from __future__ import annotations

import pytest

from src.viz.style import CATEGORICAL, INK, ORDINAL_BLUE

OFF_WHITE = "#fcfcfb"  # the design system's surface, replaced by the page's


def _srgb(h: str) -> list[float]:
    h = h.lstrip("#")
    return [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]


def _linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _luminance(h: str) -> float:
    r, g, b = (_linear(c) for c in _srgb(h))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: str, b: str) -> float:
    la, lb = _luminance(a), _luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def test_surface_is_the_page_white():
    """a printed figure sits on white paper. An off-white panel behind it
    reads as a grey box that encodes nothing."""
    assert INK["surface"] == "#ffffff"


@pytest.mark.parametrize(
    "token", sorted(set(CATEGORICAL) | set(INK.values()) | set(ORDINAL_BLUE))
)
def test_no_token_loses_contrast_against_the_white_surface(token):
    """The surface substitution may not loosen the contrast gate. It cannot,
    because a lighter surface raises the ratio for every colour darker than
    it, but the claim is in the module docstring so it is tested."""
    if token == INK["surface"]:
        return
    assert _contrast(token, INK["surface"]) >= _contrast(token, OFF_WHITE)


def test_relief_rule_tokens_are_the_ones_documented():
    """Three slots sit below 3:1 and therefore oblige direct labels. If a
    palette edit changes which, the docstring's relief-rule note is stale."""
    low = {c for c in CATEGORICAL if _contrast(c, INK["surface"]) < 3.0}
    assert low == {"#1baf7a", "#eda100", "#e87ba4"}


def test_no_figure_prints_a_raw_model_key():
    """A reader who meets `dfm_robust` in a figure and `Factor model (robust)`
    in the table beside it has to work out that they are the same thing.

    This renders every figure and inspects its drawn text, because the labels
    are set in a dozen places and a substitution is easy to miss in one.
    """
    import pathlib
    import re

    import matplotlib

    matplotlib.use("Agg")
    from src.labels import MODEL_NAMES
    from src.viz import figures

    keys = set(MODEL_NAMES)
    offenders: dict[str, list[str]] = {}
    original = figures._save

    def spy(fig, name):
        drawn: list[str] = []
        for ax in fig.get_axes():
            drawn += [t.get_text() for t in ax.texts]
            legend = ax.get_legend()
            if legend is not None:
                drawn += [t.get_text() for t in legend.get_texts()]
            drawn += [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
        raw = sorted(
            {
                w
                for t in drawn
                for w in re.findall(r"[a-z0-9]+(?:_[a-z0-9]+)+", t or "")
                if w in keys
            }
        )
        if raw:
            offenders[name] = raw

    # build_all writes into results/figures. A test must not overwrite the
    # artifacts the study is built from, so the spy renders each figure and
    # discards it instead of calling through to the real _save.
    def discard(fig, name):
        spy_collect(fig, name)
        import matplotlib.pyplot as plt

        plt.close(fig)
        return pathlib.Path(name)

    spy_collect = spy
    figures._save = discard
    try:
        figures.build_all()
    finally:
        figures._save = original
    assert not offenders, f"figures printing raw model keys: {offenders}"
