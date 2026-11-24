"""Every figure in the study, each drawn from an artifact under ``results/``.

One function per figure, all sharing ``src.viz.style``. Presentation only:
nothing here computes a result, so a figure cannot disagree with the table it
is drawn from.

Form choices are recorded next to each figure because the form is the first
decision and the easiest to get wrong. The recurring one is EMPHASIS: several
of these compare eighteen to twenty-two models, which is far past the
eight-hue ceiling, so the models that carry the argument take palette slots in
order and the rest go to the de-emphasis grey. Cycling or generating hues past
the ceiling is never the answer.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.labels import model_name
from src.viz.style import (
    CATEGORICAL,
    CATEGORICAL_ALLPAIRS4,
    INK,
    MARKERS,
    STATUS,
    apply_style,
    emphasis_colors,
    end_labels,
    finish,
    legend_below,
    relayout_labels,
    series_kw,
)

TABLES = Path("results/tables")
FIGURES = Path("results/figures")
COVID = (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-06-30"))


def _save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    relayout_labels(fig)
    fig.savefig(path)
    plt.close(fig)
    return path


# Bradley's liberal interval for a 5% nominal level. The size figures draw it
# because the text classifies a procedure by whether its rejection rate falls
# inside, and a figure showing only the 5% rule invites the reader to judge
# against a line the study does not use.
TOLERANCE = (2.5, 7.5)


def _tolerance_band(ax, nominal, horizontal=True):
    """Shade the interval the text treats as correctly sized."""
    lo, hi = TOLERANCE
    span = ax.axhspan if horizontal else ax.axvspan
    span(lo, hi, color=INK["grid"], alpha=0.55, zorder=0, linewidth=0)
    rule = ax.axhline if horizontal else ax.axvline
    rule(nominal, color=INK["muted"], linewidth=1.0, zorder=2)


def _errors(results_dir: str = "results", target_path: str | None = None):
    """Answered errors, wide by model, with the ablation variant excluded."""
    from src.eval import metrics as M

    preds = M.load_predictions(results_dir)
    preds = preds[~preds["variant"].astype(str).str.contains("balanced")]
    target = M.load_target() if target_path is None else M.load_target(target_path)
    prepared = M.prepare(preds, target)
    answered = prepared[prepared["yhat"].notna()]
    return prepared, answered.pivot_table(
        index=["nowcast_date", "target_quarter"], columns="model", values="error"
    )


# ---------------------------------------------------------------------------
# F1 -- the ragged edge
# ---------------------------------------------------------------------------
def fig_ragged_edge(nowcast_date="2015-05-12", window_days=210) -> Path:
    """FORM: emphasis on a timeline, one row per series.

    The previous version of this figure was a binary observed/missing raster.
    A raster spends its whole visual budget on the cells that ARE observed,
    which is the uninteresting 95%, and forces the reader to find the edge by
    scanning for where the blue stops. The edge is the subject, so it gets the
    marks: each series is a hairline track ending in a dot at its last
    observation, and the rows are ordered by publication lag, which turns the
    ragged edge into a staircase the reader sees at once.

    One series, one colour (slot 1) -- colouring rows by lag magnitude would
    be a value-ramp on nominal categories, double-encoding the very distance
    the x-axis already shows.
    """
    from src.data import alfred_client, info_set
    from src.experiments.run_backtest import load_config

    apply_style()
    cfg = load_config("configs/data.yaml")
    series = alfred_client.panel_series_ids(cfg, "baseline")
    t = pd.Timestamp(nowcast_date)
    quarter = t.to_period("Q").start_time
    panel = info_set.information_set(
        t, series, target_id=cfg["target"]["fred_id"], target_quarter=quarter
    )
    window = panel.loc[t - pd.Timedelta(days=window_days) : t]

    rows = []
    for col in window.columns:
        last = window[col].last_valid_index()
        if last is None:
            continue
        rows.append(
            {
                "series": col,
                "last_obs": last,
                "lag_days": (t - last).days,
                "n_obs": int(window[col].notna().sum()),
            }
        )
    edge = pd.DataFrame(rows).sort_values("lag_days")

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    y = np.arange(len(edge))
    x0 = mdates.date2num(window.index[0])
    for yi, (_, r) in zip(y, edge.iterrows()):
        xe = mdates.date2num(r["last_obs"])
        ax.plot(
            [x0, xe],
            [yi, yi],
            color=CATEGORICAL[0],
            linewidth=1.6,
            solid_capstyle="butt",
            alpha=0.55,
            zorder=2,
        )
        # the missing stretch: the ragged edge itself, drawn as absence
        ax.plot(
            [xe, mdates.date2num(t)],
            [yi, yi],
            color=INK["grid"],
            linewidth=1.6,
            solid_capstyle="butt",
            zorder=1,
        )
        ax.plot(
            xe,
            yi,
            marker="o",
            markersize=4.6,
            color=CATEGORICAL[0],
            markeredgecolor=INK["surface"],
            markeredgewidth=1.2,
            zorder=3,
        )

    ax.axvline(mdates.date2num(t), color=STATUS["critical"], linewidth=1.6, zorder=4)
    ax.axvline(mdates.date2num(quarter), color=INK["muted"], linewidth=1.0, zorder=4)
    top = len(edge) - 0.35
    ax.annotate(
        f"nowcast date {t.date()}",
        (mdates.date2num(t), top),
        xytext=(-5, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=9.0,
        color=INK["secondary"],
    )
    ax.annotate(
        f"target quarter begins {quarter.date()}",
        (mdates.date2num(quarter), top),
        xytext=(-5, -12),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=9.0,
        color=INK["muted"],
    )
    # Selective direct labels, one per DISTINCT lag rather than per row: the
    # lags cluster into a few groups because releases follow a shared monthly
    # calendar, and that clustering is the substantive content. A label on
    # every row would repeat the same three numbers twenty-two times.
    lags = edge["lag_days"].to_numpy()
    for lag, group in edge.groupby("lag_days"):
        idx = int(np.where(lags == lag)[0].max())
        ax.annotate(
            f"{int(lag)} days  ({len(group)} series)",
            (mdates.date2num(group["last_obs"].iloc[0]), idx),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=9.0,
            color=INK["secondary"],
            va="center",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(edge["series"], fontsize=9.0)
    ax.set_ylim(-1, len(edge))
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    handles = [
        plt.Line2D(
            [],
            [],
            color=CATEGORICAL[0],
            linewidth=1.8,
            marker="o",
            markersize=4.6,
            markeredgecolor=INK["surface"],
            label="observed in this vintage",
        ),
        plt.Line2D(
            [],
            [],
            color=INK["grid"],
            linewidth=1.8,
            label="not yet published at the nowcast date",
        ),
    ]
    finish(
        ax,
        xlabel="observation date",
        title="Publication lags on one nowcast date: the ragged edge",
        subtitle=(
            f"{len(edge)} baseline indicators ordered by publication lag; "
            "the target itself is unobserved for the whole quarter"
        ),
        grid="x",
    )
    ax.tick_params(axis="y", length=0)
    legend_below(ax, handles=handles, ncol=2, y=-0.13)
    return _save(fig, "F1_ragged_edge.png")


# ---------------------------------------------------------------------------
# F2 -- empirical size against the overlap
# ---------------------------------------------------------------------------
def _reps_label(frame) -> str:
    """ "N replications", taken from the artifact.

    These subtitles carried a hand-typed "2,000 replications" and stayed
    there when the study moved to 20,000, so the figures contradicted the
    tables they were drawn from.
    """
    reps = sorted({int(r) for r in frame["n_reps"].dropna().unique()})
    if len(reps) != 1:
        raise ValueError(f"expected one replication count, got {reps}")
    return f"{reps[0]:,} replications"


def fig_inference_size() -> Path:
    """FORM: multi-line, four series, categorical -- the series ARE the
    subject (which test), and the reader must tell them apart.

    Four series is the point at which direct labels become mandatory, so each
    line is labelled at its right end and each carries a distinct marker as
    secondary encoding for greyscale printing. The nominal level is a
    reference rule in muted ink, not a fifth series.
    """
    apply_style()
    t10 = pd.read_csv(TABLES / "T10_inference_size.csv")
    # The overlap DGP only: it is the one where m varies and everything else
    # is held fixed, which is what makes the curve attributable. The other
    # three data-generating processes are the subject of F9.
    a = t10[
        t10["part"].str.startswith("A")
        & (t10["n_quarters"] == 82)
        & (t10["comparison"] == "overlap")
    ].sort_values("dates_per_quarter")
    if a.empty:
        raise ValueError(
            "no Panel A 'overlap' rows in T10_inference_size.csv; the size "
            "study's DGP labels have changed and this filter is stale"
        )
    nominal = 100 * float(t10["nominal_size"].iloc[0])

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    series = [
        ("weekly_hac4", "weekly DM, HAC(4)", "HAC(4)"),
        ("weekly_hac8", "weekly DM, HAC(8)", "HAC(8)"),
        ("weekly_hac17", "weekly DM, HAC(17)", "HAC(17)"),
        ("quarterly_t", "quarter-aggregated $t$-test", "quart. $t$"),
        ("quarterly_sign", "quarter-aggregated sign test", "sign"),
    ]
    x = a["dates_per_quarter"].to_numpy()
    for slot, (col, label, short) in enumerate(series):
        yv = 100 * a[col].to_numpy()
        ax.plot(x, yv, label=label, zorder=3, **series_kw(slot))
        # A legend carries identity (the method requires one at two or more
        # series); the end label is the supplement, kept short so five of
        # them fit in the right margin without crowding.
        ax.annotate(
            short,
            (x[-1], yv[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=9.0,
            color=INK["secondary"],
            va="center",
        )

    _tolerance_band(ax, nominal, horizontal=True)
    # Anchored at the left edge and below the rule: at x[1] the label sat on
    # top of the curves that cross there.
    ax.annotate(
        f"nominal {nominal:.0f}%",
        (x[0], nominal),
        xytext=(2, -12),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=9.0,
        color=INK["muted"],
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.0f}" for v in x])
    ax.set_xlim(x[0] * 0.85, x[-1] * 2.9)
    ax.set_ylim(0, max(45, 105 * a["weekly_hac4"].max()))
    finish(
        ax,
        xlabel="nowcast dates aimed at the same target quarter (log scale)",
        ylabel="rejection rate at 5% nominal (%)",
        title="Excess rejection tracks the within-target overlap",
        subtitle=("exact Gaussian null, mean zero, 82 quarters, " + _reps_label(t10)),
        grid="y",
    )
    legend_below(ax, ncol=3, y=-0.15)
    return _save(fig, "F2_inference_size.png")


# ---------------------------------------------------------------------------
# F3 -- required sample against effect size
# ---------------------------------------------------------------------------
def fig_power() -> Path:
    """FORM: single curve, a reference rule, and the comparisons placed on it.

    One series, so no legend box -- the axis labels and title name what is
    plotted. Each observed comparison is drawn at the sample it would need,
    which is a point ON the curve, so the comparisons separate vertically and
    every label sits beside its own mark. An earlier version put them all on
    the study's own sample line, where five marks a few thousandths apart
    shared one horizontal baseline and their labels had to be stacked into a
    column that no longer touched the marks it named.
    """
    apply_style()
    t11 = pd.read_csv(TABLES / "T11_power.csv")
    grid = t11[t11["kind"] == "effect_size_grid"].dropna(
        subset=["n_quarters_for_80pct_power"]
    )
    obs = t11[t11["kind"] == "observed_comparison"]
    n_obs = int(obs["n_quarters"].max())

    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.plot(
        grid["p_hat"],
        grid["n_quarters_for_80pct_power"],
        color=CATEGORICAL[0],
        marker="o",
        markersize=4.5,
        markeredgecolor=INK["surface"],
        markeredgewidth=1.2,
        zorder=3,
        label="_nolegend_",
    )

    # Everything above the rule needs more target quarters than exist here.
    ax.axhspan(n_obs, 1e5, color=INK["grid"], alpha=0.45, zorder=0)
    ax.axhline(n_obs, color=STATUS["critical"], linewidth=1.4, zorder=2)

    # Comparisons at the same effect size need the same sample, so they share
    # a point; |p - 1/2| is what sets it, hence the fold.
    groups: dict[tuple[float, float], list[str]] = {}
    off_grid: list[str] = []
    for _, r in obs.iterrows():
        if pd.isna(r["p_hat"]):
            continue
        name = model_name(r["comparison"].split(" vs ")[0], short=True)
        need = r["n_quarters_for_80pct_power"]
        if pd.isna(need):
            off_grid.append(name)
            continue
        groups.setdefault(
            (round(max(r["p_hat"], 1 - r["p_hat"]), 4), float(need)), []
        ).append(name)

    # Points near the left edge have no room for a label outside the axes, so
    # the side is chosen from the space available rather than fixed.
    x_lo = float(grid["p_hat"].min())
    x_hi = float(grid["p_hat"].max())
    flip_below = x_lo + 0.25 * (x_hi - x_lo)
    for (p_eff, need), names in sorted(groups.items()):
        ax.plot(
            p_eff,
            need,
            marker="v",
            markersize=7.0,
            color=CATEGORICAL[1],
            markeredgecolor=INK["surface"],
            markeredgewidth=1.3,
            zorder=5,
        )
        # The curve falls to the right, so the space to the LEFT of a point on
        # it is empty at that height: labels go there and cannot collide with
        # the curve or with each other.
        right = p_eff < flip_below
        ax.annotate(
            "\n".join(names),
            (p_eff, need),
            xytext=(10 if right else -10, 0),
            textcoords="offset points",
            ha="left" if right else "right",
            va="center",
            fontsize=8.5,
            color=INK["secondary"],
        )

    if off_grid:
        ax.annotate(
            ", ".join(off_grid) + ": past the end of the grid",
            (grid["p_hat"].max(), grid["n_quarters_for_80pct_power"].max()),
            xytext=(0, -2),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=8.5,
            color=INK["secondary"],
        )
    ax.annotate(
        f"this study: {n_obs} target quarters",
        (grid["p_hat"].max(), n_obs),
        xytext=(0, 6),
        textcoords="offset points",
        ha="right",
        fontsize=9.0,
        color=INK["secondary"],
    )
    ax.set_yscale("log")
    ax.set_ylim(
        grid["n_quarters_for_80pct_power"].min() / 1.6,
        grid["n_quarters_for_80pct_power"].max() * 1.6,
    )
    finish(
        ax,
        xlabel="P(lower squared error in a target quarter)",
        ylabel="target quarters required (log scale)",
        title="What it would take: sample required for 80% power",
        subtitle=(
            "exact two-sided sign test, alpha = 0.05; shaded region is "
            "beyond this study's sample"
        ),
        grid="y",
    )
    return _save(fig, "F3_power.png")


# ---------------------------------------------------------------------------
# F4 -- RMSE against the fraction of quarters won
# ---------------------------------------------------------------------------
def fig_mean_vs_median() -> Path:
    """FORM: scatter with quadrant rules -- the reader must locate entities in
    a two-dimensional space, and the finding IS a quadrant.

    An all-pairs form, so colour is capped at three slots: the models in the
    "better RMSE yet behind in most quarters" quadrant take slot 2, those
    significantly behind on the sign test take slot 8's red only through the
    status token (they are a state, not an identity), and everything else is
    de-emphasis grey. This is the figure the study's second finding needs and
    which no relative-RMSE table can show.
    """
    apply_style()
    t7 = pd.read_csv(TABLES / "T7_vs_naive_benchmark.csv")
    t7 = t7.assign(pct_ahead=100 * t7["quarters_ahead"] / t7["n_quarters"])

    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    ax.axvline(1.0, color=INK["axis"], linewidth=0.9, zorder=1)
    ax.axhline(50.0, color=INK["axis"], linewidth=0.9, zorder=1)
    # The quadrant that carries the finding, washed not outlined. axvspan's
    # ymin/ymax are AXES FRACTIONS, not data coordinates, so using 0.5 there
    # shades to half the panel height rather than to the 50% rule -- which
    # put `dfm` visually outside the quadrant it belongs to. fill_between
    # works in data coordinates and is correct by construction.
    ax.fill_between(
        [t7["rmse_rel_naive"].min() - 0.03, 1.0],
        0,
        50.0,
        color=CATEGORICAL[1],
        alpha=0.07,
        zorder=0,
        linewidth=0,
    )

    candidates: list[tuple[int, pd.Series]] = []
    for _, r in t7.iterrows():
        in_quadrant = r["rmse_rel_naive"] < 1 and r["pct_ahead"] < 50
        sig_behind = r["p_sign"] < 0.05 and r["pct_ahead"] < 50
        if sig_behind:
            colour, marker, size, z = STATUS["critical"], "X", 8.0, 4
        elif in_quadrant:
            colour, marker, size, z = CATEGORICAL[1], "o", 6.5, 3
        else:
            colour, marker, size, z = INK["deemph"], "o", 5.0, 2
        ax.plot(
            r["rmse_rel_naive"],
            r["pct_ahead"],
            marker=marker,
            markersize=size,
            color=colour,
            markeredgecolor=INK["surface"],
            markeredgewidth=1.3,
            zorder=z,
        )
        if sig_behind or r["rmse_rel_naive"] < 0.83 or r["rmse_rel_naive"] > 1.15:
            candidates.append((0 if sig_behind else 1, r))

    # The completed-panel arms sit within 0.02 of each other on both axes, so
    # labelling every candidate stacked four names on one point and made all
    # of them unreadable. Labels are placed in priority order and a candidate
    # whose mark is too close to one already labelled is left as a point; the
    # families it belongs to are named in the caption instead.
    x_span = t7["rmse_rel_naive"].max() - t7["rmse_rel_naive"].min()
    y_span = t7["pct_ahead"].max() - t7["pct_ahead"].min()
    placed: list[tuple[float, float]] = []
    for _, r in sorted(candidates, key=lambda c: c[0]):
        x, y = r["rmse_rel_naive"], r["pct_ahead"]
        too_close = any(
            abs(x - px) < 0.05 * x_span and abs(y - py) < 0.08 * y_span
            for px, py in placed
        )
        if too_close:
            continue
        placed.append((x, y))
        # A centred label is ambiguous here: these names are wide enough that
        # half of one reaches back past the neighbouring mark, and the reader
        # anchors on the end of the text rather than its middle. Each label
        # therefore BEGINS at its own mark, or ends at it where beginning
        # there would run off the panel.
        near_right = x > t7["rmse_rel_naive"].max() - 0.18
        ax.annotate(
            model_name(r["model"], short=True),
            (x, y),
            xytext=(-7 if near_right else 7, -3),
            textcoords="offset points",
            ha="right" if near_right else "left",
            va="top",
            fontsize=8.5,
            color=INK["secondary"],
        )

    # The quadrant's meaning belongs in the caption, not inside the panel: a
    # paragraph of text competes with the marks for attention, and the counts
    # it would state are checked against the artifact by the claim tests.
    handles = [
        plt.Line2D(
            [],
            [],
            marker="X",
            linestyle="",
            markersize=7.5,
            color=STATUS["critical"],
            markeredgecolor=INK["surface"],
            label="behind in most quarters, significantly (sign test)",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=6,
            color=CATEGORICAL[1],
            markeredgecolor=INK["surface"],
            label="behind in most quarters, not significantly",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markersize=5,
            color=INK["deemph"],
            markeredgecolor=INK["surface"],
            label="other models",
        ),
    ]
    # The legend sat over the cloud in the upper right. A legend box on top of
    # the marks it describes is worse than one that costs a little height.
    legend_below(ax, handles=handles, ncol=1, y=-0.15)

    # The quadrant carries the finding, so it says what it is and how many
    # models are in it. The count comes from the artifact, not from the text.
    n_quadrant = int(((t7["rmse_rel_naive"] < 1) & (t7["pct_ahead"] < 50)).sum())
    x0 = t7["rmse_rel_naive"].min() - 0.03
    y0 = t7["pct_ahead"].min() - 3
    ax.annotate(
        f"lower RMSE, yet behind in\nmost quarters: {n_quadrant} models",
        (x0 + 0.012, y0 + 1.2),
        ha="left",
        va="bottom",
        fontsize=9.0,
        color=INK["secondary"],
        zorder=5,
    )
    ax.set_ylim(t7["pct_ahead"].min() - 3, t7["pct_ahead"].max() + 4)
    ax.set_xlim(t7["rmse_rel_naive"].min() - 0.03, t7["rmse_rel_naive"].max() + 0.04)
    finish(
        ax,
        xlabel="RMSE relative to the expanding-mean benchmark",
        ylabel="target quarters with lower squared error (%)",
        title="RMSE and the typical quarter disagree",
        grid="both",
    )
    return _save(fig, "F4_mean_vs_median.png")


# ---------------------------------------------------------------------------
# F5 -- where the advantage over a constant is earned
# ---------------------------------------------------------------------------
def fig_crisis_concentration(results_dir="results") -> Path:
    """FORM: small multiples, emphasis, cumulative line.

    Plotting the cumulative squared-error advantage over the trivial
    benchmark shows at a glance that a flat curve jumps at two dates and is
    flat again afterwards. A bar chart per quarter would show the same
    numbers and hide the point, because the eye cannot integrate eighty bars.

    Two panels rather than one: 2020Q2 is an order of magnitude larger than
    2008Q4, so on a shared axis the earlier episode is a flat line and the
    paper's claim that BOTH episodes contribute cannot be checked. The right
    panel is the same series truncated before 2020, on its own scale, which
    is the honest way to show two effects of very different size -- never a
    second y-axis.
    """
    apply_style()
    _, wide = _errors(results_dir)
    models = [m for m in ("dfm", "dfm_robust", "xgboost") if m in wide.columns]
    if not models or "naive_mean" not in wide.columns:
        raise RuntimeError("need the naive benchmark and at least one model")

    curves = {}
    for model in models:
        pair = wide[[model, "naive_mean"]].dropna()
        q = pair.index.get_level_values("target_quarter")
        adv = (pair["naive_mean"] ** 2 - pair[model] ** 2).groupby(q).mean()
        curves[model] = adv.sort_index().cumsum()

    fig, axes = plt.subplots(
        1, 2, figsize=(7.8, 3.9), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    cutoff = pd.Timestamp("2020-01-01")
    panels = [
        (axes[0], None, "Whole sample", [("2008-07-01", "2009-07-01"), COVID]),
        (axes[1], cutoff, "Before 2020, own scale", [("2008-07-01", "2009-07-01")]),
    ]
    for ax, cut, title, spans in panels:
        ends = []
        for slot, model in enumerate(models):
            c = curves[model]
            if cut is not None:
                c = c[c.index < cut]
            ax.plot(
                c.index,
                c.to_numpy(),
                label=model_name(model),
                zorder=3,
                **series_kw(slot, marker=False),
            )
            ends.append((c.index[-1], float(c.iloc[-1]), model_name(model, short=True)))
        end_labels(ax, ends, min_gap_pt=13.0)
        ax.axhline(0, color=INK["axis"], linewidth=0.9, zorder=1)
        for a, b in spans:
            ax.axvspan(
                pd.Timestamp(a), pd.Timestamp(b), color=INK["grid"], alpha=0.7, zorder=0
            )
        ax.set_title(title, loc="left", pad=6, fontsize=8.8)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        # Two panels side by side leave each about three inches; at a
        # four-year tick the year labels ran into one another, so the ticks
        # thin out and the labels shrink a step.
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", labelsize=9.0)
        last = max(
            (c[c.index < cut] if cut is not None else c).index[-1]
            for c in curves.values()
        )
        ax.set_xlim(right=last + pd.Timedelta(days=760))
    axes[0].set_ylabel("cumulative MSE advantage over a constant")
    for ax in axes:
        ax.set_xlabel("target quarter")
    fig.suptitle(
        "The advantage over a constant is earned in two episodes",
        x=0.005,
        y=1.06,
        ha="left",
        fontsize=9.5,
    )
    fig.text(
        0.005,
        1.005,
        "flat stretches are quarters where the model and the benchmark tie; "
        "shaded bands are the two crisis episodes",
        fontsize=9.0,
        color=INK["muted"],
        ha="left",
    )
    # The legend spans both panels, so it belongs to the figure. Hung off one
    # axes it was counted by tight_layout as part of that panel, which pushed
    # the panels apart and left an empty column between them: each plot used
    # well under half the width it was given.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        handlelength=2.0,
        borderpad=0.3,
        columnspacing=1.6,
        fontsize=9.0,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=2.5)
    return _save(fig, "F5_crisis_concentration.png")


# ---------------------------------------------------------------------------
# F6 -- the publication cycle, per model
# ---------------------------------------------------------------------------
def fig_cycle_profile() -> Path:
    """FORM: emphasis over a balanced panel of eighteen models.

    Eighteen series is far past the eight-hue ceiling. The four models that
    span the finding -- the best cycle exploiter, the trivial benchmark as the
    zero-gain control, and the two arms that get WORSE -- take slots in order;
    the remaining fourteen are the de-emphasis grey and carry the context.
    """
    apply_style()
    grid = pd.read_csv(TABLES / "F5_cycle_profile_balanced.csv", index_col=0)
    summary = pd.read_csv(TABLES / "F5_within_quarter_cycle.csv").set_index("model")
    weeks = sorted((int(c) for c in grid.columns), reverse=True)

    highlight = [
        m
        for m in (
            "dfm",
            "midas",
            "chronos_ragged_edge",
            "naive_mean",
            "chronos_aggregate_q",
        )
        if m in grid.index
    ]
    colours = emphasis_colors(grid.index, highlight)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for model in grid.index:
        y = [grid.loc[model, str(w)] for w in weeks]
        is_hi = model in highlight
        ax.plot(
            weeks,
            y,
            color=colours[model],
            linewidth=2.0 if is_hi else 0.9,
            alpha=1.0 if is_hi else 0.75,
            zorder=3 if is_hi else 1,
            marker=MARKERS[highlight.index(model)] if is_hi else None,
            markersize=4.0 if is_hi else 0,
            markeredgecolor=INK["surface"],
            markeredgewidth=1.1,
            label=model_name(model) if is_hi else None,
        )
        if is_hi:
            gain = summary.loc[model, "gain_to_best_pct"]
            ax.annotate(
                f"{model_name(model, short=True)}  {gain:+.0f}%",
                (weeks[-1], y[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=9.0,
                color=INK["secondary"],
                va="center",
            )
    ax.invert_xaxis()
    ax.set_xlim(weeks[0] + 0.5, weeks[-1] - 5.5)
    # Cycle weeks are integers; matplotlib's default locator produces half-week
    # ticks on this range, which name a quantity that does not exist.
    ax.set_xticks([w for w in weeks if w % 2 == 1 or w == 0])
    ax.set_xticklabels([str(w) for w in weeks if w % 2 == 1 or w == 0])
    handles = [
        plt.Line2D(
            [],
            [],
            color=CATEGORICAL[i],
            linewidth=2.0,
            marker=MARKERS[i],
            markersize=4.5,
            markeredgecolor=INK["surface"],
            label=model_name(m),
        )
        for i, m in enumerate(highlight)
    ] + [
        plt.Line2D(
            [],
            [],
            color=INK["deemph"],
            linewidth=1.0,
            label=f"other models ({len(grid.index) - len(highlight)})",
        )
    ]
    finish(
        ax,
        xlabel="weeks before the end of the target quarter (0 = quarter end)",
        ylabel="RMSE (pp, annualised)",
        title="What another week of data buys is a property of the model",
        subtitle=(
            "balanced panel: identical (model, quarter) cells at every cycle week; "
            "labels give the gain to each model's best week"
        ),
        grid="y",
    )
    legend_below(ax, handles=handles, ncol=3, y=-0.16)
    return _save(fig, "F6_cycle_profile.png")


# ---------------------------------------------------------------------------
# F7 -- the fine-tuning null
# ---------------------------------------------------------------------------
def fig_finetune() -> Path:
    """FORM: dot plot of deviations against a zero rule, not a line chart.

    The previous version plotted RMSE against step count as a single line and
    showed four points, because the two fine-tuning modes overlapped at every
    budget and the line hid it. The finding is "nothing moves", so the right
    form makes the null measurable: every configuration is a dot on a common
    axis of deviation from zero-shot, with the +-0.2% band drawn, so the reader
    sees both that the deviations are tiny and that they are non-monotone.
    """
    apply_style()
    t12 = pd.read_csv(TABLES / "T12_finetune_budget.csv")
    zero = float(t12.loc[t12["mode"] == "zero-shot", "rmse"].iloc[0])
    ft = t12[t12["mode"] != "zero-shot"].copy()
    ft["label"] = ft["mode"].map({"lora": "LoRA", "full": "full weights"})
    ft = ft.sort_values(["label", "finetune_steps"])

    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    band = float(ft["deviation_pct"].abs().max())
    ax.axvspan(-band, band, color=CATEGORICAL[0], alpha=0.07, zorder=0)
    ax.axvline(0, color=STATUS["critical"], linewidth=1.4, zorder=2)

    labels, ys = [], []
    for i, (_, r) in enumerate(ft.iterrows()):
        slot = 0 if r["label"] == "LoRA" else 1
        ax.plot(
            r["deviation_pct"],
            i,
            marker=MARKERS[slot],
            markersize=6.5,
            color=CATEGORICAL[slot],
            markeredgecolor=INK["surface"],
            markeredgewidth=1.3,
            zorder=4,
        )
        ax.annotate(
            f"{r['deviation_pct']:+.2f}%",
            (r["deviation_pct"], i),
            xytext=(9 if r["deviation_pct"] >= 0 else -9, 0),
            textcoords="offset points",
            ha="left" if r["deviation_pct"] >= 0 else "right",
            va="center",
            fontsize=9.0,
            color=INK["secondary"],
        )
        labels.append(f"{r['label']}, {int(r['finetune_steps'])} steps")
        ys.append(i)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9.0)
    ax.set_ylim(-0.8, len(ft) - 0.2)
    ax.set_xlim(-band * 3.2, band * 3.2)
    ax.annotate(
        f"zero-shot RMSE {zero:.4f}",
        (0, len(ft) - 0.4),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=9.0,
        color=INK["secondary"],
        va="center",
    )
    handles = [
        plt.Line2D(
            [],
            [],
            marker=MARKERS[i],
            linestyle="",
            markersize=6,
            color=CATEGORICAL[i],
            markeredgecolor=INK["surface"],
            label=name,
        )
        for i, name in enumerate(["LoRA", "full weights"])
    ]
    ax.legend(handles=handles, loc="lower right", handlelength=1.2, borderpad=0.3)
    finish(
        ax,
        xlabel="deviation from zero-shot RMSE (%)",
        title="Fine-tuning does not move the result, by either route",
        subtitle=(
            f"every configuration lies within \u00b1{band:.2f}% of zero-shot, "
            "and the ordering is non-monotone in the budget"
        ),
        grid="x",
    )
    ax.tick_params(axis="y", length=0)
    return _save(fig, "F7_finetune.png")


# ---------------------------------------------------------------------------
# F8 -- the two crisis episodes
# ---------------------------------------------------------------------------
def fig_crisis_episodes(results_dir="results") -> Path:
    """FORM: small multiples, emphasis within each panel.

    Two panels rather than one, because the episodes are an order of
    magnitude apart in scale and a shared axis would flatten 2008 into a line.
    The realised value is the reference and wears primary ink, not a series
    hue: it is not one of the things being compared.

    The DFM's line stops in 2008Q4: it abstains through all of 2009 because
    its divergence screen rejects every candidate specification there. The
    truncation is real and is left visible rather than interpolated over.
    """
    apply_style()
    prepared, _ = _errors(results_dir)
    crisis = prepared[
        prepared["episode"].isin(["gfc", "covid"]) & prepared["yhat"].notna()
    ]
    if crisis.empty:
        raise RuntimeError("no crisis-episode predictions")

    highlight = ["dfm", "chronos_ragged_edge", "naive_mean"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8))
    for ax, episode, title in zip(
        axes, ["gfc", "covid"], ["2008–09 financial crisis", "2020–21 pandemic"]
    ):
        sub = crisis[crisis["episode"] == episode]
        if sub.empty:
            continue
        by_q = sub.groupby(["target_quarter", "model"])["yhat"].mean().unstack()
        actual = sub.groupby("target_quarter")["actual"].first()
        colours = emphasis_colors(by_q.columns, highlight)
        for model in by_q.columns:
            is_hi = model in highlight
            ax.plot(
                by_q.index,
                by_q[model],
                color=colours[model],
                linewidth=1.8 if is_hi else 0.8,
                alpha=1.0 if is_hi else 0.7,
                zorder=3 if is_hi else 1,
                marker=MARKERS[highlight.index(model)] if is_hi else None,
                markersize=4.0 if is_hi else 0,
                markeredgecolor=INK["surface"],
                markeredgewidth=1.1,
                label=model_name(model) if is_hi else None,
            )
        ax.plot(
            actual.index,
            actual.to_numpy(),
            color=INK["primary"],
            linewidth=2.2,
            zorder=4,
            label="realised (second estimate)",
        )
        # The thin grey tracks carried no legend entry, so a reader had no
        # way to learn what they were.
        ax.plot(
            [],
            [],
            color=INK["deemph"],
            linewidth=1.0,
            label="other models",
        )
        ax.set_title(title, loc="left", pad=6)
        ax.tick_params(axis="x", rotation=45)
        # strftime has no quarter directive, so "%Y Q%q" silently yields
        # "2008 Q" for every quarter -- eight identical, useless labels.
        # The quarter is computed and set explicitly instead.
        ticks = list(by_q.index)
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [f"{d.year} Q{(d.month - 1) // 3 + 1}" for d in ticks],
            rotation=45,
            ha="right",
        )
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("GDP growth (pp, annualised)")
    # Figure-level legend, for the reason given in fig_crisis_concentration:
    # attached to the right-hand axes it dragged that panel across the page.
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=5,
        handlelength=1.8,
        borderpad=0.3,
        columnspacing=1.4,
        fontsize=9.0,
    )
    fig.suptitle(
        "Where the models diverge: mean nowcast per target quarter",
        x=0.005,
        ha="left",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.5)
    return _save(fig, "F8_crisis_episodes.png")


# ---------------------------------------------------------------------------
# F9 -- what each procedure survives, and what it does not
# ---------------------------------------------------------------------------
def fig_size_by_dgp() -> Path:
    """FORM: Cleveland dot plot -- eight procedures against four departures
    from the ideal case. Rows are the entities being judged, grouped by what
    each procedure's validity argument assumes; the four marks in a row are
    the conditions.

    A dot plot rather than grouped bars because the quantity is a rate at a
    reference level, not a magnitude from zero: what the reader must see is
    each mark's distance from the nominal 5% rule, and bars would encode the
    distance from zero instead. The four marks share a row, so every pair of
    hues is adjacent and CATEGORICAL_ALLPAIRS4 is used rather than consecutive
    slots -- consecutive slots fail the all-pairs gate.

    Identity is carried by the legend plus a distinct marker per condition.
    The aqua's sub-3:1 contrast is relieved by the table view (tab:size in the
    paper), which is why no in-chart labels duplicate the legend here: an
    earlier version labelled the top row directly and the labels collided.
    """
    apply_style()
    t10 = pd.read_csv(TABLES / "T10_inference_size.csv")
    a = t10[t10["part"].str.startswith("A") & (t10["n_quarters"] == 82)]

    # One row per DGP at the study's own overlap, so each column differs from
    # the first in exactly one feature.
    dgps = [
        ("overlap", "overlap only", 17.0),
        ("serial", "+ serial dependence", 17.0),
        ("heavy_tail", "+ outlier cluster", 17.0),
        ("unbalanced", "+ unbalanced grid", None),
    ]
    # Grouped top to bottom by what the procedure acknowledges. A blank slot
    # separates the groups and carries the group name, which keeps the class
    # out of the legend: it is a property of the row, not of a series.
    groups = [
        ("acknowledges nothing", [("weekly_hac4", "weekly DM, HAC(4)")]),
        (
            "acknowledges the cluster",
            [
                ("cluster_cr1", "cluster-robust CR1"),
                ("wild_bootstrap", "wild cluster bootstrap"),
                ("quarterly_sign", "exact sign test"),
            ],
        ),
        (
            "allows dependence between clusters, asymptotically in $G$",
            [
                ("hac_quarterly", "HAC on cluster means"),
                ("moving_block", "moving-block bootstrap"),
            ],
        ),
        (
            "better finite-sample reference distribution",
            [
                ("ibragimov_muller_q4", "Ibragimov\u2013M\u00fcller, $q=4$"),
                ("hac_fixed_b025", "fixed-$b$ HAC, $b=0.25$"),
            ],
        ),
    ]
    nominal = 100 * float(t10["nominal_size"].iloc[0])

    # Lay the rows out from the bottom up, leaving 1.1 slots between groups.
    rows: list[tuple[str, str, float]] = []
    heads: list[tuple[str, float]] = []
    pos = 0.0
    for name, members in reversed(groups):
        for col, label in reversed(members):
            rows.append((col, label, pos))
            pos += 1.0
        heads.append((name, pos - 0.42))
        pos += 1.1

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    _tolerance_band(ax, nominal, horizontal=False)

    for k, (dgp, label, m) in enumerate(dgps):
        sub = a[a["comparison"] == dgp]
        # `m` disambiguates the overlap sweep, which has four rows. For the
        # unbalanced process the recorded value is a realised average rather
        # than a setting, so it moves between runs and cannot be matched on.
        if m is not None:
            sub = sub[sub["dates_per_quarter"] == m]
        if sub.empty:
            raise ValueError(f"no Panel A rows for DGP {dgp!r} at m={m}")
        r = sub.iloc[0]
        # At nominal all four conditions land on the same value, so a surface
        # ring alone is not enough to keep them countable: each condition is
        # offset by a fixed fraction of a row. The offset is in y, which
        # carries no quantity here, so it distorts nothing the reader reads.
        dy = (k - 0.5 * (len(dgps) - 1)) * 0.11
        ax.plot(
            [100 * float(r[col]) for col, _, _ in rows],
            [y + dy for _, _, y in rows],
            linestyle="none",
            marker=MARKERS[k],
            markersize=8.5,
            color=CATEGORICAL_ALLPAIRS4[k],
            markeredgecolor=INK["surface"],
            markeredgewidth=1.6,
            label=label,
            zorder=4,
        )

    for name, y in heads:
        ax.annotate(
            name,
            (0.0, y),
            xytext=(2, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.5,
            style="italic",
            color=INK["muted"],
        )

    ax.annotate(
        f"nominal {nominal:.0f}%",
        (nominal, -0.72),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9.0,
        color=INK["muted"],
    )
    ax.set_yticks([y for _, _, y in rows])
    ax.set_yticklabels([lab for _, lab, _ in rows])
    ax.set_ylim(-1.0, pos - 0.9)
    ax.set_xlim(0, 50)

    finish(
        ax,
        xlabel="rejection rate at 5% nominal (%)",
        ylabel="",
        title="Only two procedures survive serial dependence",
        subtitle=(
            "exact Gaussian null, 82 target quarters, 17 nowcast dates each, "
            + _reps_label(t10)
        ),
        grid="x",
    )
    legend_below(ax, ncol=4, y=-0.18)
    return _save(fig, "F9_size_by_dgp.png")


# ---------------------------------------------------------------------------
# F10 -- the recommendation's boundary condition
# ---------------------------------------------------------------------------
def fig_im_by_groups() -> Path:
    """FORM: multi-line over an ordered tuning parameter, one line per
    data-generating process -- the subject is a RELATIONSHIP (how size varies
    with q) that differs by regime, which is what a line carries and a dot
    plot does not.

    This is the figure that bounds the study's own recommendation. Under
    short-range dependence the Ibragimov--Muller test is flat in q and at
    nominal, so the tuning choice does not matter. Under long-range
    dependence it is monotone increasing in q, because contiguous groups stop
    being approximately independent and adding groups adds units that are
    assumed independent and are not. The practical rule -- keep q small when
    persistence may be long-range -- is the slope of the orange line.

    Five series, so the legend carries identity and only the lines that
    finish clear of the nominal rule are also labelled directly; the aqua's
    sub-3:1 contrast is relieved by one of those labels.
    """
    apply_style()
    t10 = pd.read_csv(TABLES / "T10_inference_size.csv")
    a = t10[t10["part"].str.startswith("A")]
    im_cols = sorted(
        (c for c in a.columns if c.startswith("ibragimov_muller_q") and "rot" not in c),
        key=lambda c: int(c.rsplit("q", 1)[1]),
    )
    if not im_cols:
        raise ValueError("no Ibragimov-Muller columns in T10_inference_size.csv")
    qs = [int(c.rsplit("q", 1)[1]) for c in im_cols]
    nominal = 100 * float(t10["nominal_size"].iloc[0])

    # One line per process, in the order the study introduces them. The
    # overlap row at the study's own grid density is the control.
    series = [
        ("overlap", 17.0, "overlap only"),
        ("serial", 17.0, "serial, $\\rho=0.4$"),
        ("serial_strong", 17.0, "serial, $\\rho=0.7$"),
        ("long_memory", 17.0, "near unit root"),
        ("regime", 17.0, "regime break"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    _tolerance_band(ax, nominal, horizontal=True)
    ends = []
    for slot, (dgp, m, label) in enumerate(series):
        sub = a[(a["comparison"] == dgp) & (a["dates_per_quarter"] == m)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        yv = [100 * float(r[c]) for c in im_cols]
        ax.plot(qs, yv, label=label, zorder=3, **series_kw(slot))
        # Direct labels are SELECTIVE: with five series the legend carries
        # identity, and the two lines that finish within a point of the
        # nominal rule cannot be labelled there without colliding with it and
        # with each other. They are the two whose message is "flat at
        # nominal", which the rule itself already conveys.
        if abs(yv[-1] - nominal) >= 1.2:
            ends.append((qs[-1], yv[-1], label))

    end_labels(ax, ends, min_gap_pt=13.0)
    ax.annotate(
        f"nominal {nominal:.0f}%",
        (qs[0], nominal),
        xytext=(0, -14),
        textcoords="offset points",
        ha="left",
        fontsize=9.0,
        color=INK["muted"],
    )
    ax.set_xticks(qs)
    ax.set_xlim(qs[0] - 0.3, qs[-1] + 3.2)
    ax.set_ylim(0, None)
    finish(
        ax,
        xlabel="number of contiguous groups $q$",
        ylabel="rejection rate at 5% nominal (%)",
        title="The recommendation holds for short-range dependence, not for long",
        subtitle=(
            "Ibragimov\u2013M\u00fcller test, exact Gaussian null, 82 target "
            "quarters, " + _reps_label(t10)
        ),
        grid="y",
    )
    legend_below(ax, ncol=3, y=-0.16)
    return _save(fig, "F10_im_by_groups.png")


# ---------------------------------------------------------------------------
# F11 -- the evaluation design, and what it does to the sample
# ---------------------------------------------------------------------------
def fig_design_schematic() -> Path:
    """FORM: schematic timeline above a real series.

    This is the figure the study opens with, and its job is definitional
    rather than evidential: a reader outside nowcasting has to see what "a
    dense evaluation grid" is before any rejection rate means anything. The
    top panel is drawn, not measured. The bottom panel is the same structure
    in the actual loss differentials, so the claim that the weekly series is
    the quarterly series repeated is shown and not asserted.

    Colour is the emphasis form: one target quarter is highlighted through
    both panels so the eye can carry the seventeen-to-one mapping from the
    schematic into the data. Everything else is the de-emphasis grey.
    """
    apply_style()
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(7.8, 4.9), gridspec_kw={"height_ratios": [0.85, 1.15]}
    )

    # -- (a) the grid, drawn --------------------------------------------
    hi = CATEGORICAL[0]
    n_show, per_q = 5, 17
    focus = 2
    for q in range(n_show):
        y = n_show - 1 - q
        start = q * 13  # a new target quarter every 13 weeks
        weeks = np.arange(start, start + per_q)
        colour = hi if q == focus else INK["deemph"]
        z = 4 if q == focus else 2
        ax0.plot(
            weeks,
            np.full(per_q, y),
            linestyle="none",
            marker="o",
            markersize=4.2,
            color=colour,
            markeredgecolor=INK["surface"],
            markeredgewidth=0.8,
            zorder=z,
        )
        # the realised value: published once, after the last nowcast date
        ax0.plot(
            [start + per_q + 1],
            [y],
            linestyle="none",
            marker="D",
            markersize=7.5,
            color=colour,
            markeredgecolor=INK["surface"],
            markeredgewidth=1.2,
            zorder=z + 1,
        )
        ax0.plot(
            [weeks[0], start + per_q + 1],
            [y, y],
            color=colour,
            linewidth=0.8,
            zorder=z - 1,
        )
    ax0.annotate(
        "17 nowcast dates",
        (focus * 13, n_show - 1 - focus),
        xytext=(-10, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=9.0,
        color=INK["secondary"],
    )
    ax0.annotate(
        "one realised value",
        (2 * 13 + per_q + 1, n_show - 1 - focus),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=9.0,
        color=INK["secondary"],
    )
    ax0.set_yticks(range(n_show))
    ax0.set_yticklabels([f"quarter {n_show - i}" for i in range(n_show)])
    ax0.set_xlim(-24, (n_show - 1) * 13 + per_q + 22)
    ax0.set_ylim(-0.8, n_show - 0.2)
    ax0.set_xticks([])
    finish(
        ax0,
        xlabel="",
        ylabel="",
        title="Each target quarter accumulates many forecasts, scored against one value",
        subtitle="schematic; a quarter is still being backcast while the next is already being nowcast",
        grid="none",
    )

    # -- (b) the same structure in the data ------------------------------
    _, wide = _errors()
    m1, m2 = "dfm_robust", "naive_mean"
    pair = wide[[m1, m2]].dropna().sort_index(level="target_quarter")
    d = (pair[m1] ** 2 - pair[m2] ** 2).to_numpy(float)
    q = pd.factorize(pair.index.get_level_values("target_quarter"))[0]
    # A window rather than the whole series: one pandemic quarter is three
    # orders of magnitude larger than the rest, so on the full scale every
    # other quarter collapses onto the zero line and the block structure this
    # panel exists to show becomes invisible. That concentration is the
    # subject of the next figure; here the subject is replication.
    n_show_q = 10
    keep = q < n_show_q
    d_w, q_w = d[keep], q[keep]
    x = np.arange(len(d_w))
    ax1.plot(
        x,
        d_w,
        linestyle="none",
        marker="o",
        markersize=3.2,
        color=INK["deemph"],
        zorder=2,
        label="weekly loss differential",
    )
    for k in np.unique(q_w):
        sel = q_w == k
        ax1.plot(
            [x[sel][0], x[sel][-1]],
            [d_w[sel].mean()] * 2,
            color=hi,
            linewidth=2.4,
            zorder=4,
            label="target-quarter mean" if k == 0 else None,
        )
    ax1.set_xlim(-4, len(d_w) + 4)
    finish(
        ax1,
        xlabel=f"weekly observations, first {n_show_q} target quarters",
        ylabel="loss differential",
        title="In the data, the weekly series is the quarterly series repeated",
        subtitle=(
            "factor model against the expanding mean, intra-cluster "
            f"correlation 0.75; the full {len(d):,} observations span "
            f"{q.max() + 1} quarters"
        ),
        grid="y",
    )
    legend_below(ax1, ncol=2, y=-0.30)
    fig.tight_layout()
    return _save(fig, "F11_design.png")


# ---------------------------------------------------------------------------
# F12 -- where the observations go
# ---------------------------------------------------------------------------
def fig_sample_collapse() -> Path:
    """FORM: paired dot plot and a concentration curve.

    The study's central claim is a sequence of three numbers that get
    smaller. Left panel: the two counts that are sample sizes, per
    comparison, on a log scale because they span two orders of magnitude.
    Right panel: why the third quantity is not a sample size, as the
    cumulative share of the variance carried by the quarters in rank order.

    Two hues only, in slots 0 and 1, plus de-emphasis grey for the
    comparisons that are context.
    """
    apply_style()
    t16 = pd.read_csv(TABLES / "T16_effective_sample.csv")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.8, 4.2))

    # -- (a) 1,403 -> 82 -> G_eff ---------------------------------------
    # Bars on a LINEAR scale, deliberately. A log axis would make the three
    # counts comparable and legible, which is the opposite of the point: the
    # third bar has to be almost invisible beside the first.
    # The comparison the panel is about is NAMED, not inferred from a sort.
    # `sort_values("n_pairs").iloc[-1]` happened to return a foundation-model
    # arm against the expanding mean (1,403 pairs, 79.9 effective quarters)
    # while the subtitle and the caption both said bridge against factor
    # model, which is the comparison with the largest dependence adjustment
    # (815 pairs, 43.8 effective quarters) and therefore the one the panel
    # exists to show.
    REF = ("bridge", "dfm_robust")
    match = t16[(t16["model"] == REF[0]) & (t16["benchmark"] == REF[1])]
    if match.empty:
        raise RuntimeError(f"T16 has no row for {REF[0]} vs {REF[1]}")
    ref = match.iloc[0]
    counts = [
        ("forecast pairs", float(ref["n_pairs"]), 0),
        ("independent target quarters", float(ref["n_clusters"]), 1),
        ("after the dependence adjustment", float(ref["ess_serial"]), 2),
    ]
    ypos = np.arange(len(counts))[::-1]
    for (lab, val, slot), y in zip(counts, ypos):
        ax0.barh(
            y,
            val,
            height=0.52,
            color=CATEGORICAL[slot],
            edgecolor=INK["surface"],
            linewidth=1.2,
            zorder=3,
        )
        ax0.annotate(
            f"{val:,.0f}",
            (val, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=10.0,
            color=INK["primary"],
        )
    ax0.set_yticks(ypos)
    ax0.set_yticklabels([lab for lab, _, _ in counts])
    ax0.set_xlim(0, float(ref["n_pairs"]) * 1.22)
    ax0.set_ylim(-0.6, len(counts) - 0.4)
    finish(
        ax0,
        xlabel="count",
        ylabel="",
        title="Two of the three counts are sample sizes",
        subtitle="bridge equation against the factor model",
        grid="x",
    )

    # -- (b) the third is concentration ----------------------------------
    _, wide = _errors()
    for slot, (m1, m2, lab) in enumerate(
        [
            ("dfm_robust", "bridge", "factor model vs bridge"),
            ("chronos_completed_dfm", "dfm_robust", "edge completion vs factor model"),
        ]
    ):
        pair = wide[[m1, m2]].dropna().sort_index(level="target_quarter")
        d = (pair[m1] ** 2 - pair[m2] ** 2).to_numpy(float)
        g = pd.factorize(pair.index.get_level_values("target_quarter"))[0]
        dq = np.bincount(g, weights=d) / np.bincount(g)
        w = (dq - dq.mean()) ** 2
        w = np.sort(w / w.sum())[::-1]
        ax1.plot(
            np.arange(1, len(w) + 1),
            100 * np.cumsum(w),
            label=lab,
            zorder=3,
            **series_kw(slot, marker=False),
        )
    # the line a design with equal contributions would trace
    n = 82
    ax1.plot(
        np.arange(1, n + 1),
        100 * np.arange(1, n + 1) / n,
        color=INK["muted"],
        linewidth=1.0,
        linestyle="--",
        zorder=2,
        label="equal contributions",
    )
    ax1.set_xlim(0, n)
    ax1.set_ylim(0, 102)
    finish(
        ax1,
        xlabel="target quarters, ranked by contribution",
        ylabel="cumulative share of the variance (%)",
        title="The third is a concentration, not a count",
        subtitle="one quarter carries 66% and 81% of it",
        grid="y",
    )
    legend_below(ax1, ncol=1, y=-0.22)
    fig.tight_layout()
    return _save(fig, "F12_sample_collapse.png")


# ---------------------------------------------------------------------------
# F13 -- what the recommended procedures cost
# ---------------------------------------------------------------------------
def fig_power_adjusted() -> Path:
    """FORM: multi-line power curves, emphasis on the two recommended.

    Six procedures would be six hues past the all-pairs gate. The question
    is what the two recommended procedures cost against the best available,
    so those three take hues and the rest are the grey context band.
    """
    apply_style()
    pw = pd.read_csv(TABLES / "T19_power.csv")
    # The artifact now carries two processes. The figure is about what the
    # recommended procedures cost under the dependence the data exhibits, so it
    # plots the serial rows; the heavy-tail rows are in the table beside it.
    if "process" in pw.columns:
        pw = pw[pw["process"] == "serial"].reset_index(drop=True)
    mu = pw["mu"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for col, lab in (
        ("cluster_cr1_adj", None),
        ("moving_block_adj", None),
        ("quarterly_sign_adj", "other size-adjusted procedures"),
    ):
        ax.plot(
            mu, 100 * pw[col], color=INK["deemph"], linewidth=1.4, zorder=2, label=lab
        )
    ends = []
    for slot, (col, lab) in enumerate(
        [
            ("wild_bootstrap_adj", "wild cluster bootstrap"),
            ("hac_fixed_b025_adj", "fixed-$b$ HAC"),
            ("ibragimov_muller_q4_adj", "Ibragimov–Müller, $q=4$"),
        ]
    ):
        yv = 100 * pw[col].to_numpy(float)
        ax.plot(mu, yv, label=lab, zorder=4, **series_kw(slot))
        ends.append((mu[-1], yv[-1], lab))
    end_labels(ax, ends, min_gap_pt=12.0)
    ax.set_xlim(0, mu[-1] * 1.42)
    ax.set_ylim(0, 100)
    finish(
        ax,
        xlabel="mean shift $\\mu$, in within-quarter standard deviations",
        ylabel="size-adjusted power (%)",
        title="The procedures that hold their level detect less",
        subtitle=(
            "serial dependence at $\\rho=0.4$, 82 target quarters, "
            "each test at its own 5% critical value"
        ),
        grid="y",
    )
    legend_below(ax, ncol=2, y=-0.20)
    return _save(fig, "F13_power.png")


# ---------------------------------------------------------------------------
# F14 -- the single-factor experiment, as a diagram
# ---------------------------------------------------------------------------
def fig_single_factor() -> Path:
    """FORM: a drawn comparison of two pipelines, not a chart.

    The identification argument of the study is that the two arms
    share everything except one step. That is a statement about structure,
    and prose makes the reader hold five stages in mind to check it. Drawn,
    the shared stages line up and the one that differs is the only place the
    two rows are not identical.

    Two hues, in slots 0 and 1, for the two arms; shared stages take the
    de-emphasis grey so that colour marks exactly what differs.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 3.6))

    shared = INK["deemph"]
    arms = [
        (
            "Kalman filter arm",
            CATEGORICAL[0],
            1.0,
            "the model's own Kalman filter\nfills the unpublished months",
        ),
        (
            "Foundation-model arm",
            CATEGORICAL[1],
            0.0,
            "a pretrained model fills\nthem beforehand",
        ),
    ]
    stages = ["vintage panel\n(identical)", "", "same factor model", "nowcast"]
    # The stage that differs carries two lines of prose and needs the width
    # for them; at a uniform 0.95 the text ran outside its own box.
    xs = [0.0, 1.10, 2.85, 4.05]
    widths = [0.95, 1.60, 0.95, 0.95]

    for name, colour, y, treat in arms:
        for k, (x, label) in enumerate(zip(xs, stages)):
            differs = k == 1
            w = widths[k]
            txt = treat if differs else label
            # The differing stage is a light wash of the hue with a heavy
            # border in it, not a solid fill with reversed type: white on the
            # orange is 3.2:1, below the floor for text, and a solid block
            # also loses its identity in greyscale where the border survives.
            ax.add_patch(
                plt.Rectangle(
                    (x, y - 0.30),
                    w,
                    0.60,
                    facecolor=colour if differs else INK["surface"],
                    alpha=0.12 if differs else 1.0,
                    edgecolor="none" if differs else shared,
                    linewidth=1.6,
                    zorder=3,
                )
            )
            ax.add_patch(
                plt.Rectangle(
                    (x, y - 0.30),
                    w,
                    0.60,
                    facecolor="none",
                    edgecolor=colour if differs else shared,
                    linewidth=2.2 if differs else 1.6,
                    zorder=4,
                )
            )
            ax.annotate(
                txt,
                (x + w / 2, y),
                ha="center",
                va="center",
                fontsize=8.6,
                zorder=5,
                color=INK["primary"],
            )
            if k < len(xs) - 1:
                ax.annotate(
                    "",
                    xy=(xs[k + 1], y),
                    xytext=(x + w, y),
                    arrowprops={"arrowstyle": "->", "color": shared, "lw": 1.1},
                    zorder=2,
                )
        # Identity comes from a mark beside the text, never from the type:
        # a coloured heading is unreadable as a hue and illegible as a label.
        ax.add_patch(
            plt.Rectangle(
                (xs[0], y + 0.46),
                0.10,
                0.13,
                facecolor=colour,
                edgecolor="none",
                zorder=4,
            )
        )
        ax.annotate(
            name,
            (xs[0] + 0.17, y + 0.52),
            ha="left",
            va="center",
            fontsize=9.5,
            color=INK["primary"],
            zorder=4,
        )

    # what the design holds fixed, and what it varies
    ax.annotate(
        "held fixed",
        (xs[0] + widths[0] / 2, -0.62),
        ha="center",
        va="top",
        fontsize=9.0,
        color=INK["secondary"],
    )
    ax.annotate(
        "the only difference",
        (xs[1] + widths[1] / 2, -0.62),
        ha="center",
        va="top",
        fontsize=9.0,
        color=INK["primary"],
    )
    ax.annotate(
        "held fixed",
        (xs[2] + widths[2] / 2, -0.62),
        ha="center",
        va="top",
        fontsize=9.0,
        color=INK["secondary"],
    )

    ax.set_xlim(-0.12, xs[-1] + widths[-1] + 0.12)
    ax.set_ylim(-1.05, 1.75)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    finish(
        ax,
        xlabel="",
        ylabel="",
        title="The two arms differ in one step and nothing else",
        subtitle=(
            "both use the same vintage panel and the same factor model; "
            "only the treatment of the ragged edge changes"
        ),
        grid="none",
    )
    fig.tight_layout()
    return _save(fig, "F14_single_factor.png")


# Figures generated for the replication package but deliberately NOT in the
# body of the study. The referee's judgement was that the write-up carried
# too many secondary results competing with its spine, so the publication-cycle
# profile and the fine-tuning null were reduced to prose and their figures kept
# here. Membership of this set is a decision; a
# figure absent from both the study and this set is an omission, and
# tests/test_tables_integrity.py fails on it.
# F6 returned to the body of the study when that paper gained
# figures; F7 stays out.
PACKAGE_ONLY = frozenset({"F7"})

ALL_FIGURES = {
    "F2": fig_inference_size,
    "F9": fig_size_by_dgp,
    "F10": fig_im_by_groups,
    "F11": fig_design_schematic,
    "F12": fig_sample_collapse,
    "F13": fig_power_adjusted,
}


def build_all(only=None) -> dict[str, str]:
    """Build every figure, reporting per-figure failures rather than aborting."""
    out: dict[str, str] = {}
    for key, fn in ALL_FIGURES.items():
        if only and key not in only:
            continue
        try:
            out[key] = str(fn())
        except Exception as exc:  # noqa: BLE001 - one bad figure must not stop the rest
            out[key] = f"FAILED: {type(exc).__name__}: {exc}"
    return out
