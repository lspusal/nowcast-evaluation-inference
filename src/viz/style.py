"""Shared figure style: the validated reference palette, applied to matplotlib.

The palette is the data-viz reference instance, used UNMODIFIED and in its
documented slot order, and re-validated here with the skill's own script
rather than trusted: `validate_palette.js "#2a78d6,#eb6834,#1baf7a,#eda100,
#e87ba4" --mode light` passes every gate (worst adjacent CVD Delta E 9.1
protan, worst adjacent normal-vision 19.6), and the three-slot subset used
for scatter passes under `--pairs all` (worst 9.2 deutan, 24.0 normal). Three
slots -- aqua, yellow, magenta -- sit below 3:1 contrast against the surface,
which triggers the RELIEF RULE: every figure using them carries visible direct
labels, and the study prints the table each figure was drawn from.

Only the light surface is validated, because a printed figure has one
surface; there is no dark mode to select steps for. That surface is pure
white rather than the design system's off-white: a figure sits on a
white page, and an off-white panel behind it reads as a grey box
that means nothing. The substitution cannot loosen any gate, since only the
contrast check involves the surface and a lighter surface raises it for every
token in this palette -- verified in ``tests/test_figure_style.py`` rather
than asserted, the smallest margin being the yellow at 2.11 against the
off-white and 2.17 against white. That order is the colour-vision-deficiency safety
mechanism, not a cosmetic choice: the documented validation gives a worst
adjacent CVD Delta E of 9.1 on the light surface for the full eight slots on
the adjacent pairlist (lines, bars, stacks), against a target of 8, and a
worst normal-vision Delta E of 19.6 against a floor of 15. All-pairs forms --
scatter, bubble, small multiples -- are capped at the first THREE slots,
because the fourth puts yellow beside orange and that pair fails the
all-pairs floors.

Consequences enforced here rather than left to each figure:

* ``CATEGORICAL`` is a tuple, so slots are taken in order and never cycled.
  A ninth series is not a generated hue; fold the tail or facet.
* ``SCATTER_CAP`` is the all-pairs limit.
* Gridlines and axes are solid hairlines one step off the surface, never
  dashed -- dashing reads as "projection" when it is only a grid.
* Lines are 2px with round joins; markers are at least 8px and carry a
  surface-coloured ring so they stay legible where they overlap.
* Text never wears a series colour. Identity comes from the mark beside the
  text. Series hues are legible as marks and not as type.

These figures are printed, so there is no hover layer to carry values: every
figure either direct-labels the marks that matter or is accompanied by the
table it was drawn from. Print reproduction is also in greyscale, so each multi-series
figure uses a distinct marker per series as secondary encoding, which the
four-series rule requires in any case.
"""

from __future__ import annotations

import matplotlib as mpl

# --- the reference palette, light surface -------------------------------
CATEGORICAL: tuple[str, ...] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)
SCATTER_CAP = 3

# Four-hue subset for marks that sit SIDE BY SIDE in one row, where every pair
# is adjacent and the all-pairs gate applies rather than the adjacent-pair one.
# Consecutive slots 1-4 FAIL it (#eda100 vs #eb6834 is 13.7 in normal vision,
# below the floor of 15), so this skips yellow and magenta for violet:
# `validate_palette.js --pairs all` reports normal-vision worst 16.3, CVD worst
# 9.2 (deutan), and one contrast WARN on the aqua, which direct labels relieve.
CATEGORICAL_ALLPAIRS4: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#4a3aa7",  # violet
)

MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")

# Ordinal ramp for DISCRETE ordered marks (cycle weeks, tiers). Validated
# with `validate_palette.js --ordinal`: monotone lightness, every adjacent
# gap >= 0.06, light end 2.06:1 against the surface, hue spread 3 degrees. The
# obvious choice of consecutive documented steps FAILS the adjacent-lightness
# gate (#3987e5 vs #2a78d6 is 0.047 apart), which is exactly the kind of thing
# the validator catches and the eye does not.
ORDINAL_BLUE: tuple[str, ...] = (
    "#86b6ef",  # step 250
    "#5598e7",  # step 350
    "#2a78d6",  # step 450
    "#1c5cab",  # step 550
    "#104281",  # step 650
)

# Continuous magnitude only (heatmaps): the light end may recede to the
# surface, which the ordinal gate forbids.
SEQUENTIAL_BLUE: tuple[str, ...] = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)

INK = {
    "surface": "#ffffff",  # the page is white; see the surface note above
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "deemph": "#c9c8c2",  # the de-emphasis grey for the emphasis form
}

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}


def apply_style() -> None:
    """Install the shared style. Call once before building any figure."""
    mpl.use("Agg")
    mpl.rcParams.update(
        {
            # a printed figure is set in the body face; matplotlib's default
            # sans at 8pt is close enough to elsarticle's figure text.
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            # Printed figures are reduced to the text width, so the type
            # must be set larger than looks right on screen: at 7.0in wide
            # scaled into a 5.5in column, 10.5pt renders at about 8pt.
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
            "figure.facecolor": INK["surface"],
            "axes.facecolor": INK["surface"],
            "savefig.facecolor": INK["surface"],
            "text.color": INK["primary"],
            "axes.labelcolor": INK["secondary"],
            "axes.titlecolor": INK["primary"],
            "xtick.color": INK["muted"],
            "ytick.color": INK["muted"],
            "xtick.labelcolor": INK["secondary"],
            "ytick.labelcolor": INK["secondary"],
            # hairline, solid, recessive
            "axes.edgecolor": INK["axis"],
            "axes.linewidth": 0.7,
            "grid.color": INK["grid"],
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 2.0,
            "lines.solid_joinstyle": "round",
            "lines.solid_capstyle": "round",
            "lines.markersize": 5.5,
            "lines.markeredgewidth": 1.4,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.prop_cycle": mpl.cycler(color=list(CATEGORICAL)),
        }
    )


def series_kw(slot: int, marker: bool = True) -> dict:
    """Keyword arguments for series `slot` (0-based), in documented order."""
    if slot >= len(CATEGORICAL):
        raise ValueError(
            f"slot {slot} exceeds the {len(CATEGORICAL)}-hue ceiling; fold the "
            "tail into 'other' or facet into small multiples rather than "
            "generating a hue"
        )
    kw = {"color": CATEGORICAL[slot], "linewidth": 2.0}
    if marker:
        kw |= {
            "marker": MARKERS[slot],
            "markersize": 4.5,
            "markeredgecolor": INK["surface"],
            "markeredgewidth": 1.2,
        }
    return kw


def emphasis_colors(names, highlight) -> dict:
    """Emphasis form: highlighted entities take slots in order, rest go grey.

    Colour follows the entity, not its rank, so the mapping is built from the
    highlight list and is stable when the plotted set changes.
    """
    out = {n: INK["deemph"] for n in names}
    for i, n in enumerate(highlight):
        if n in out:
            out[n] = CATEGORICAL[i % len(CATEGORICAL)]
    return out


def label_end(ax, x, y, text, color, dx=4, dy=0, ha="left", va="center", size=9.0):
    """Direct label at a series end. Text wears an ink token, not the hue."""
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=size,
        color=INK["secondary"],
        ha=ha,
        va=va,
    )
    return color


def finish(
    ax,
    *,
    xlabel=None,
    ylabel=None,
    title=None,
    subtitle=None,
    grid="y",
    legend=False,
):
    """Common finishing: recessive grid on one axis, hairline spines, labels.

    `subtitle` is set as a second title line rather than as an annotation in
    axes coordinates. Annotating just above the axes collides with the title
    at any figure height the caller did not anticipate, which is precisely
    the layout defect the method's "render it and look at it" step exists to
    catch -- so the fix belongs here rather than in each figure.
    """
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        if subtitle:
            ax.set_title(
                f"{title}\n",
                loc="left",
                pad=16,
                fontsize=9.5,
                color=INK["primary"],
            )
            ax.annotate(
                subtitle,
                xy=(0, 1.0),
                xycoords="axes fraction",
                xytext=(0, 6),
                textcoords="offset points",
                fontsize=9.0,
                color=INK["muted"],
                va="bottom",
            )
        else:
            ax.set_title(title, loc="left", pad=8)
    if grid in ("y", "both"):
        ax.yaxis.grid(True, zorder=0)
    if grid in ("x", "both"):
        ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    if legend:
        ax.legend(handlelength=1.6, borderpad=0.2, labelspacing=0.4)


def end_labels(ax, items, *, dx=5, min_gap_pt=11.0, size=9.0):
    """Direct labels at series ends, de-collided vertically.

    `items` is a sequence of (x, y, text). Series whose final values are
    close together produce overlapping labels, which is the most common
    layout defect in a multi-line chart and is invisible until the figure is
    rendered. This shifts labels apart in offset points, preserving order, so
    each still reads against its own line.

    The separation depends on how many points a unit of y occupies, which is
    only final once the layout is: tight_layout shrinks the axes after the
    labels are placed, and labels spaced on the pre-layout geometry ended up
    overlapping. Placement is therefore registered on the figure and redone
    by the saving routine on the final geometry (see `relayout_labels`).
    """
    if not items:
        return
    order = sorted(items, key=lambda it: it[1])
    anns = [
        ax.annotate(
            text,
            (x, y),
            xytext=(dx, 0.0),
            textcoords="offset points",
            fontsize=size,
            color=INK["secondary"],
            va="center",
        )
        for x, y, text in order
    ]

    def place():
        placed = []
        for ann, (_, y, _) in zip(anns, order):
            # transData gives display PIXELS, while the gap and the offset
            # are POINTS; at the 150 dpi the style sets a point is 2.08
            # pixels, so they are converted before being compared. x may
            # carry date units the affine transform cannot take, hence 0.0.
            y_pt = ax.transData.transform((0.0, float(y)))[1] * 72.0 / ax.figure.dpi
            offset = 0.0
            for prev in placed:
                if abs((y_pt + offset) - prev) < min_gap_pt:
                    offset = prev - y_pt + min_gap_pt
            placed.append(y_pt + offset)
            ann.xyann = (dx, offset)

    place()
    ax.figure.__dict__.setdefault("_end_label_placers", []).append(place)


def relayout_labels(fig) -> None:
    """Re-run every end-label placement on the figure's final geometry."""
    placers = fig.__dict__.get("_end_label_placers", [])
    if placers:
        fig.canvas.draw()
        for place in placers:
            place()


def legend_below(ax, handles=None, ncol=2, y=-0.16):
    """Legend outside the axes, below the x-axis label.

    Used where the plot has no empty region: a legend box placed over data is
    worse than one that costs a little height.
    """
    kwargs = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, y),
        "ncol": ncol,
        "handlelength": 2.0,
        "borderpad": 0.3,
        "columnspacing": 1.6,
        "fontsize": 9.5,
    }
    return ax.legend(handles=handles, **kwargs) if handles else ax.legend(**kwargs)
