"""Generated paper tables must satisfy the invariants they claim to have.

These guard the two constructions that were WRONG the first time (see-63):

* F5 compared cycle weeks across DIFFERENT sets of quarters and models, so a
  single easy quarter at the tail turned a 6% gain into an apparent 90% one.
  The fix is a balanced panel; "balanced" is a checkable property, so it is
  checked here rather than trusted.
* T7 is the trivial-benchmark table the study lacked. Its whole value comes
  from `naive_mean` being a genuine real-time floor, so the sanity conditions
  that make it a floor are asserted.

The tests read the artifacts under results/ and SKIP when a table has not
been generated, so a fresh clone with no run still passes `make check`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TAB = Path("results/tables")


def _table(name: str) -> pd.DataFrame:
    path = TAB / name
    if not path.exists():
        pytest.skip(f"{path} not generated; run `make paper-tables`")
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# F5 -- the balanced within-quarter cycle panel
# --------------------------------------------------------------------------
def test_f5_cycle_panel_is_actually_balanced():
    """Every retained cycle week must cover the SAME quarters for the SAME
    models. This is the exact property the first version of F5 violated."""
    grid = _table("F5_cycle_profile_balanced.csv").set_index("model")
    assert not grid.empty
    # No hole anywhere in the model x week grid: a NaN means some model had no
    # observation in some week, i.e. the panel is not balanced.
    assert grid.notna().all().all(), "unbalanced: missing model x week cells"
    assert len(grid.columns) >= 3, "too few cycle weeks to form a profile"


def test_f5_weeks_are_contiguous_and_run_towards_the_release():
    grid = _table("F5_cycle_profile_balanced.csv").set_index("model")
    weeks = sorted(int(c) for c in grid.columns)
    assert weeks == list(range(weeks[0], weeks[-1] + 1)), "cycle weeks have gaps"
    f5 = _table("F5_within_quarter_cycle.csv")
    # first_week is EARLIER in the cycle (larger "weeks before quarter end").
    assert (f5["first_week"] > f5["last_week"]).all()


def test_f5_naive_mean_gains_nothing_from_the_cycle():
    """The positive control. A real-time expanding mean of published GDP
    cannot see within-quarter indicator news, so its error must be flat
    across the cycle. If this ever moves, the panel construction is wrong or
    the naive benchmark is peeking."""
    f5 = _table("F5_within_quarter_cycle.csv").set_index("model")
    if "naive_mean" not in f5.index:
        pytest.skip("naive_mean not in this run")
    assert abs(f5.loc["naive_mean", "gain_to_best_pct"]) < 1.0
    assert abs(f5.loc["naive_mean", "gain_at_end_pct"]) < 1.0


def test_f5_best_week_is_consistent_with_the_grid():
    """`best_week` and the RMSEs in the summary table must be recomputable
    from the grid the figure is drawn from -- no independent derivation."""
    f5 = _table("F5_within_quarter_cycle.csv").set_index("model")
    grid = _table("F5_cycle_profile_balanced.csv").set_index("model")
    for model, row in f5.iterrows():
        series = grid.loc[model]
        assert int(series.idxmin()) == int(row["best_week"])
        assert series.min() == pytest.approx(row["rmse_best_week"], rel=1e-9)
        assert series[str(int(row["first_week"]))] == pytest.approx(
            row["rmse_first_week"], rel=1e-9
        )


def test_f5_gains_are_internally_consistent():
    f5 = _table("F5_within_quarter_cycle.csv")
    expected_best = 100 * (1 - f5["rmse_best_week"] / f5["rmse_first_week"])
    expected_end = 100 * (1 - f5["rmse_last_week"] / f5["rmse_first_week"])
    assert np.allclose(f5["gain_to_best_pct"], expected_best)
    assert np.allclose(f5["gain_at_end_pct"], expected_end)
    # The best week is by construction at least as good as the last week.
    assert (f5["gain_to_best_pct"] >= f5["gain_at_end_pct"] - 1e-9).all()


# --------------------------------------------------------------------------
# T7 -- the trivial benchmark
# --------------------------------------------------------------------------
def test_t7_relative_rmse_matches_its_components():
    t7 = _table("T7_vs_naive_benchmark.csv")
    assert np.allclose(t7["rmse_rel_naive"], t7["rmse_model"] / t7["rmse_naive"])


def test_t7_verdicts_follow_their_own_estimands():
    """T7 now carries TWO verdicts, one per estimand: mean loss decided by
    the wild cluster bootstrap and the typical quarter by the exact sign
    test. Each must be a function of its own statistic and of nothing else,
    because conflating them is how a claim about the median quarter gets
    reported as a claim about accuracy."""
    t7 = _table("T7_vs_naive_benchmark.csv")
    for _, r in t7.iterrows():
        if r["p_wild_bootstrap"] < 0.05:
            expected = (
                "lower mean loss" if r["rmse_rel_naive"] < 1 else "higher mean loss"
            )
        else:
            expected = "indistinguishable"
        assert r["verdict_mean_loss"] == expected, f"{r['model']}: mean-loss verdict"

        if r["p_sign"] < 0.05:
            ahead = r["quarters_ahead"] > r["n_quarters"] / 2
            expected = "ahead in most quarters" if ahead else "behind in most quarters"
        else:
            expected = "indistinguishable"
        assert (
            r["verdict_typical_quarter"] == expected
        ), f"{r['model']}: typical-quarter verdict"


def test_t7_the_two_estimands_can_disagree():
    """The taxonomy is only worth having if the two estimands actually come
    apart on this data. They do: at least one model is significant on one and
    not the other."""
    t7 = _table("T7_vs_naive_benchmark.csv")
    mean_sig = t7["verdict_mean_loss"] != "indistinguishable"
    quarter_sig = t7["verdict_typical_quarter"] != "indistinguishable"
    assert (mean_sig & ~quarter_sig).any() or (
        quarter_sig & ~mean_sig
    ).any(), "no model separates the two estimands; the distinction would be moot"


def test_t7_quarterly_statistic_agrees_with_the_rmse_ordering():
    """A negative quarterly t statistic must mean the model's mean squared
    error is the lower one. Sign confusion between the two is exactly the bug
    that inverted the XGBoost finding in a draft of §66."""
    t7 = _table("T7_vs_naive_benchmark.csv")
    signed = t7[t7["p_t_quarterly"] < 0.05]
    if signed.empty:
        pytest.skip("no comparison significant under quarterly inference")
    assert ((signed["t_stat_quarterly"] < 0) == (signed["rmse_rel_naive"] < 1)).all()


def test_t7_weekly_hac_is_more_permissive_than_quarterly():
    """The audit's premise: the weekly HAC statistic over-rejects on this
    design. If a comparison were ever significant quarterly but not weekly,
    the overlap story would be wrong and §66 would need revisiting."""
    t7 = _table("T7_vs_naive_benchmark.csv")
    n_weekly = int((t7["p_weekly_hac"] < 0.05).sum())
    n_quarterly = int((t7["p_t_quarterly"] < 0.05).sum())
    assert n_weekly >= n_quarterly, (
        f"{n_quarterly} comparisons significant quarterly but only {n_weekly} "
        "weekly -- the overlap argument in §66 does not hold here"
    )


def test_t8_audit_shows_bandwidth_dependence():
    """T8 exists to document that the bandwidth drives the inference.

    Widening the bandwidth need not raise every p-value -- negative
    autocorrelation at longer lags can shrink the long-run variance -- and on
    this sample the ONE exception is `bridge` vs `dfm_robust`, whose p-value
    falls from 0.121 to 0.103. That is the study's one surviving finding, and
    the fact that its significance does not come from the bandwidth is a
    point in its favour, so the test asserts the majority pattern plus the
    conclusion §66 actually draws.
    """
    t8 = _table("T8_inference_audit.csv")
    rises = int((t8["p_hac17"] >= t8["p_hac4"] - 1e-9).sum())
    assert rises >= len(t8) - 1, (
        f"only {rises}/{len(t8)} p-values rise with the bandwidth; the overlap "
        "correction is not behaving as §66 describes"
    )
    assert (t8["p_hac4"] < 0.05).sum() > (t8["p_t_quarterly"] < 0.05).sum()


def test_t8_surviving_finding_is_not_a_bandwidth_artifact():
    """Whatever survives quarterly inference must not be a comparison whose
    weekly significance depended on the short bandwidth."""
    t8 = _table("T8_inference_audit.csv").set_index(["model", "benchmark"])
    survivors = t8[(t8["p_sign"] < 0.05) | (t8["p_wilcoxon"] < 0.05)]
    if survivors.empty:
        pytest.skip("nothing survives the quarterly tests in this run")
    for key, r in survivors.iterrows():
        assert r["p_hac17"] <= 0.5, (
            f"{key} survives a quarterly test but its HAC(17) p-value is "
            f"{r['p_hac17']:.3f}; check the sign conventions agree"
        )


def test_t7_naive_benchmark_is_not_trivially_beatable_by_ar1():
    """naive_ar1 is the second trivial model. If an AR(1) on published GDP
    growth were to beat the expanding mean comfortably, the mean would be a
    straw benchmark and the whole interpretive frame of the design notes would
    be too generous. Over 2005-2024 it does not."""
    t7 = _table("T7_vs_naive_benchmark.csv").set_index("model")
    if "naive_ar1" not in t7.index:
        pytest.skip("naive_ar1 not in this run")
    assert t7.loc["naive_ar1", "rmse_rel_naive"] > 1.0


def test_t7_covers_every_model_family():
    """A trivial-benchmark table that silently omits a family would let a
    model escape the comparison that anchors the study's interpretation."""
    t7 = _table("T7_vs_naive_benchmark.csv")
    models = set(t7["model"])
    for family in ("dfm", "midas", "chronos", "timesfm", "lagllama"):
        assert any(family in m for m in models), f"no {family} arm in T7"


# --------------------------------------------------------------------------
# T6 -- nested variance decomposition
# --------------------------------------------------------------------------
def test_t6_nested_r2_is_monotone():
    """The terms are nested, so R2 cannot fall as terms are added. A drop
    would mean the design matrices are not actually nested."""
    t6 = _table("T6_variance_decomposition.csv")
    order = ["quarter", "quarter+week", "quarter+week+model", "quarter+week*model"]
    for spec, g in t6.groupby("spec"):
        g = g.set_index("term").reindex([t for t in order if t in set(g["term"])])
        r2 = g["r2"].to_numpy()
        assert np.all(np.diff(r2) >= -1e-9), f"{spec}: R2 falls when nesting"
        assert np.all((r2 >= 0) & (r2 <= 1))
        # The interaction must use more parameters than the main effects.
        assert np.all(np.diff(g["n_params"].to_numpy()) > 0)


def test_t6_quarter_effect_dominates():
    """The headline claim of the decomposition: between-quarter difficulty is
    the bulk of the variance in both specifications."""
    t6 = _table("T6_variance_decomposition.csv")
    for spec, g in t6.groupby("spec"):
        q = float(g.loc[g["term"] == "quarter", "r2"].iloc[0])
        full = float(g["r2"].max())
        assert q > 0.5, f"{spec}: quarter explains only {q:.3f}"
        # Everything else together adds little variance, which is exactly why
        # the per-model cycle gains in F5 must be reported separately.
        assert full - q < 0.10, f"{spec}: non-quarter terms add {full - q:.3f}"


# --------------------------------------------------------------------------
# T10 -- the size study that makes the inference audit a result (§69)
# --------------------------------------------------------------------------
def _size_table() -> pd.DataFrame:
    path = TAB / "T10_inference_size.csv"
    if not path.exists():
        pytest.skip(f"{path} not generated; run `make inference-size`")
    return pd.read_csv(path)


def test_t10_no_overlap_control_is_correctly_sized():
    """The control that makes §69 an argument rather than an anecdote: with
    ONE nowcast date per target quarter there is no overlap, so the weekly
    HAC(4) test must be correctly sized. If this ever fails, the distortion
    is not the overlap and §69's mechanism is wrong."""
    t10 = _size_table()
    ctrl = t10[(t10["part"].str.startswith("A")) & (t10["dates_per_quarter"] == 1.0)]
    if ctrl.empty:
        pytest.skip("no no-overlap control row")
    size = float(ctrl["weekly_hac4"].iloc[0])
    nominal = float(ctrl["nominal_size"].iloc[0])
    assert size < 3 * nominal, (
        f"weekly HAC(4) rejects {size:.1%} with NO overlap; the overlap "
        "cannot then be the cause of the distortion"
    )


def test_t10_distortion_grows_with_the_overlap():
    """Monotonicity in the overlap is what lets another study predict its own
    distortion from its design."""
    t10 = _size_table()
    a = t10[
        (t10["part"].str.startswith("A"))
        & (t10["n_quarters"] == 82)
        & (t10["comparison"] == "tau=1.0")
    ].sort_values("dates_per_quarter")
    if len(a) < 3:
        pytest.skip("not enough overlap settings")
    sizes = a["weekly_hac4"].to_numpy()
    assert sizes[-1] > 4 * sizes[0], (
        f"HAC(4) size only moves {sizes[0]:.1%} -> {sizes[-1]:.1%} across the "
        "overlap grid"
    )
    # The largest overlap must be badly oversized, or the audit has no teeth.
    assert sizes[-1] > 0.20


def test_t10_quarterly_tests_are_correctly_sized_under_pure_overlap():
    """Under the overlap DGP alone, every cluster-level procedure must hold
    its level -- that is what licenses using any of them. The claim is
    restricted to that DGP on purpose: the serial-dependence DGP breaks all
    of them, which is the finding the next test pins."""
    t10 = _size_table()
    pure = t10[t10["part"].str.startswith("A") & (t10["comparison"] == "overlap")]
    if pure.empty:
        pytest.skip("no pure-overlap rows")
    nominal = float(pure["nominal_size"].iloc[0])
    for col in ("quarterly_t", "quarterly_sign", "cluster_cr1", "wild_bootstrap"):
        if col not in pure.columns:
            continue
        worst = float(pure[col].max())
        assert (
            worst < 2 * nominal
        ), f"{col} rejects up to {worst:.1%} under pure overlap"


def test_t10_serial_dependence_breaks_every_cluster_procedure():
    """The study's sharpest methodological result: at the autocorrelation the
    data exhibits, NO procedure examined holds its nominal level on 82
    clusters. If this ever stopped holding, the study's central caution would
    be wrong and its substantive nulls would need revisiting."""
    t10 = _size_table()
    serial = t10[t10["part"].str.startswith("A") & (t10["comparison"] == "serial")]
    if serial.empty:
        pytest.skip("no serial-dependence rows")
    nominal = float(serial["nominal_size"].iloc[0])
    row = serial.iloc[0]
    for col in ("cluster_cr1", "wild_bootstrap", "quarterly_t"):
        if col in serial.columns:
            assert row[col] > 2.5 * nominal, (
                f"{col} is no longer badly oversized under serial dependence "
                f"({row[col]:.1%}); §96 would need revisiting"
            )
    # The dependence-robust procedures must be BETTER but not exact.
    for col in ("hac_quarterly", "moving_block"):
        if col in serial.columns:
            assert row[col] < row["cluster_cr1"], (
                f"{col} is not an improvement on cluster inference under "
                "serial dependence, which is its whole purpose"
            )
            assert row[col] > 1.5 * nominal, (
                f"{col} holds its level under serial dependence at G=82 "
                "after all; the study's caution is too strong"
            )


def test_t10_heavy_tails_and_imbalance_alone_do_not_break_clustering():
    """Attribution: the study claims the cluster procedures fail because of
    serial dependence, not because of tails or unbalanced grids. Those two
    DGPs must therefore leave them correctly sized."""
    t10 = _size_table()
    for dgp in ("heavy_tail", "unbalanced"):
        rows = t10[t10["part"].str.startswith("A") & (t10["comparison"] == dgp)]
        if rows.empty:
            continue
        nominal = float(rows["nominal_size"].iloc[0])
        for col in ("cluster_cr1", "wild_bootstrap", "quarterly_sign"):
            if col in rows.columns:
                worst = float(rows[col].max())
                assert worst < 2 * nominal, (
                    f"{col} rejects {worst:.1%} under the {dgp} DGP; the "
                    "paper attributes its failure to serial dependence alone"
                )


def test_t10_sign_test_is_the_only_test_valid_on_real_data():
    """§69's conclusion, asserted: on the real loss differentials the sign
    test is correctly sized while the weekly HAC test is not. The quarterly
    t-test is deliberately NOT required to be well sized here -- fat tails
    make it unable to reject at all, which is itself the finding."""
    t10 = _size_table()
    b = t10[t10["part"].str.startswith("B")]
    if b.empty:
        pytest.skip("no randomization part; needs a run on disk")
    nominal = float(b["nominal_size"].iloc[0])
    assert float(b["quarterly_sign"].max()) < 2 * nominal, (
        "the sign test is not correctly sized on real data; §69's "
        "recommendation does not follow"
    )
    assert float(b["weekly_hac4"].max()) > 4 * nominal, (
        "the weekly HAC(4) test is not oversized on any real comparison; "
        "§66's retraction would need revisiting"
    )
    # Uncalibrated, not merely inflated: the spread across pairs is the point.
    spread = float(b["weekly_hac4"].max()) - float(b["weekly_hac4"].min())
    assert spread > 0.20, f"HAC(4) size varies only {spread:.1%} across pairs"


def test_t10_surviving_finding_is_certified_by_a_calibrated_test():
    """The bridge-vs-DFM result is the study's one positive claim, and §69
    argues it is certified by the one test measured as calibrated on that
    comparison's own data. That link must hold in the artifacts."""
    t10 = _size_table()
    row = t10[t10["comparison"] == "bridge vs dfm_robust"]
    if row.empty:
        pytest.skip("bridge comparison not in the size study")
    nominal = float(row["nominal_size"].iloc[0])
    sign_size = float(row["quarterly_sign"].iloc[0])
    assert abs(sign_size - nominal) < nominal, (
        f"sign test size on the bridge comparison is {sign_size:.1%} against "
        f"a nominal {nominal:.1%}"
    )
    t8 = _table("T8_inference_audit.csv")
    bridge = t8[(t8["model"] == "bridge") & (t8["benchmark"] == "dfm_robust")]
    if bridge.empty:
        pytest.skip("bridge row absent from T8")
    assert float(bridge["p_sign"].iloc[0]) < 0.05


# --------------------------------------------------------------------------
# T11 -- power: what it would take to settle what this study cannot
# --------------------------------------------------------------------------
def _power_table() -> pd.DataFrame:
    path = TAB / "T11_power.csv"
    if not path.exists():
        pytest.skip(f"{path} not generated; run `make power`")
    return pd.read_csv(path)


def test_t11_required_sample_falls_as_the_effect_grows():
    t11 = _power_table()
    g = t11[t11["kind"] == "effect_size_grid"].sort_values("p_hat")
    need = g["n_quarters_for_80pct_power"].dropna().to_numpy()
    assert len(need) >= 4
    assert np.all(np.diff(need) <= 0), "required n does not fall with the effect"


def test_t11_power_at_the_observed_sample_is_consistent_with_the_grid():
    """The grid's power-at-82 column and the per-comparison power must agree
    about the same effect size, or one of them is computed wrongly."""
    t11 = _power_table()
    g = t11[t11["kind"] == "effect_size_grid"].sort_values("p_hat")
    # Power must rise with the effect size at fixed n.
    assert np.all(np.diff(g["power_at_82_quarters"].to_numpy()) > 0)
    assert g["power_at_82_quarters"].iloc[0] < 0.30
    assert g["power_at_82_quarters"].iloc[-1] > 0.90


def test_t11_the_significant_findings_are_the_well_powered_ones():
    """The study's internal consistency check: the comparisons that came out
    significant under the sign test must be the ones it had power for. If a
    low-power comparison were significant, that would be a red flag for a
    false positive rather than a finding."""
    t11 = _power_table()
    obs = t11[t11["kind"] == "observed_comparison"].copy()
    if obs.empty:
        pytest.skip("no observed comparisons in the power table")
    t8 = _table("T8_inference_audit.csv")
    t8 = t8.assign(comparison=t8["model"] + " vs " + t8["benchmark"])
    merged = obs.merge(t8[["comparison", "p_sign"]], on="comparison", how="inner")
    if merged.empty:
        pytest.skip("power and audit tables share no comparison")
    sig = merged[merged["p_sign"] < 0.05]
    for _, r in sig.iterrows():
        assert r["power_at_observed_n"] > 0.5, (
            f"{r['comparison']} is significant at power "
            f"{r['power_at_observed_n']:.2f} -- suspect a false positive"
        )


def test_t11_confidence_interval_brackets_the_point_estimate():
    """The optimistic bound must be a genuine lower bound: reporting only the
    point estimate would overstate how precisely the required sample is
    known, since n explodes as p approaches 0.5."""
    t11 = _power_table()
    obs = t11[t11["kind"] == "observed_comparison"].dropna(
        subset=["n_quarters_for_80pct_power", "n_quarters_optimistic"]
    )
    if obs.empty:
        pytest.skip("no comparison has both bounds")
    assert (obs["n_quarters_optimistic"] <= obs["n_quarters_for_80pct_power"]).all()
    assert (obs["p_ci_low"] <= obs["p_hat"]).all()
    assert (obs["p_ci_high"] >= obs["p_hat"]).all()


_NEEDS_VINTAGE_CACHE = frozenset({"F1"})


def test_every_figure_builds():
    """`build_all` reports per-figure failures instead of aborting, which is
    right for a batch but means a broken figure is only a line of stdout. F2
    was broken for a while that way: its filter selected `comparison ==
    "tau=1.0"`, a Panel A label that stopped existing when the size study went
    to four data-generating processes, so it silently produced nothing and the
    paper kept compiling against a stale PNG.
    """
    from src.viz import figures

    # build_all writes into results/figures. A test must not overwrite the
    # artifacts the study is built from: doing so leaves every figure newer
    # than the PDFs that already embed it, which is indistinguishable from a
    # genuinely stale build. This test only asks whether each figure draws,
    # so render and discard.
    original = figures._save

    def discard(fig, name):
        import matplotlib.pyplot as plt

        plt.close(fig)
        return Path(name)

    figures._save = discard
    try:
        built = figures.build_all()
    finally:
        figures._save = original
    failed = {k: v for k, v in built.items() if str(v).startswith("FAILED")}
    # One figure redraws a real information set from the vintage cache, so it
    # cannot be built where that cache is absent -- in the distributed
    # replication package, for instance, which ships the rendered PNG instead.
    # Everything else is drawn from the result artifacts and must build.
    if not Path("data/raw").exists():
        failed = {k: v for k, v in failed.items() if k not in _NEEDS_VINTAGE_CACHE}
    assert not failed, "figures failed to build:\n" + "\n".join(
        f"  {k}: {v}" for k, v in failed.items()
    )
    assert set(built) == set(figures.ALL_FIGURES)


def test_the_paper_includes_every_figure_that_is_built():
    """A figure that builds but is never \\includegraphics'd is dead weight,
    and one referenced but not built breaks the build. Both directions are
    checked so the figure inventory and the study cannot drift apart."""
    papers = [
        p for p in (Path("paper/paper_a.tex"), Path("paper/paper_b.tex")) if p.exists()
    ]
    if not papers:
        pytest.skip("no manuscript present")
    text = "\n".join(p.read_text() for p in papers)
    from src.viz import figures

    for key in figures.ALL_FIGURES:
        if key in figures.PACKAGE_ONLY:
            assert f"{key}_" not in text, (
                f"{key} is listed as replication-package-only but appears in "
                "the study; remove it from PACKAGE_ONLY or from the study"
            )
            continue
        assert f"{key}_" in text, (
            f"{key} is built by src/viz/figures.py but never included in the "
            f"paper; include it, add it to PACKAGE_ONLY deliberately, or drop "
            f"it from ALL_FIGURES"
        )
