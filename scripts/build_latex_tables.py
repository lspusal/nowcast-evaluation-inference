"""Generate booktabs table fragments for paper/main.tex from results/tables.

`paper/main.tex` carries no result numbers inside tables: it \\input{}s the
fragments written here, each generated from the CSV it describes. A
regenerated artifact therefore changes the typeset paper, and a stale number
cannot survive in a table.

Each entry in SPECS names the source CSV, the columns to print, per-column
formatting, printed headers, and the LaTeX label and caption. Adding a table
to the study means adding a spec here, not typing numbers into the prose.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import re

import pandas as pd

TABLES = Path("results/tables")
OUT = Path("paper/tables")

# fmt: off
SPECS: dict[str, dict] = {
    # The size table was one 17-column block scaled to the text width and
    # unreadable. It is now three: the attribution processes, the stress
    # processes, and the randomization panel, each with the columns that
    # matter for it.
    "T_size": {
        "csv": "T10_inference_size.csv",
        "where_in": {"comparison": ["overlap", "serial", "heavy_tail",
                                    "unbalanced"]},
        "cols": ["comparison", "dates_per_quarter", "weekly_hac4",
                 "weekly_hac17", "cluster_cr1", "wild_bootstrap",
                 "quarterly_sign", "hac_quarterly", "moving_block",
                 "ibragimov_muller_q4", "hac_fixed_b025"],
        "head": ["Process", "$m$", "HAC(4)", "HAC(17)", "CR1", "Wild",
                 "Sign", "HAC-$q$", "Block", "IM", "Fx-$b$"],
        "fmt": {"dates_per_quarter": "trim", "weekly_hac4": "pct1",
                "weekly_hac17": "pct1", "cluster_cr1": "pct1",
                "wild_bootstrap": "pct1", "quarterly_sign": "pct1",
                "hac_quarterly": "pct1", "moving_block": "pct1",
                "ibragimov_muller_q4": "pct1", "hac_fixed_b025": "pct1"},
        "align": "lrrrrrrrrrr",
        "label": "tab:size",
        "small": True,
    },
    # The fixed-b reading changes the bandwidth and the reference distribution
    # at once. A referee asks which repairs the size, and the answer is not the
    # expected one, so the intermediate case gets its own small table rather
    # than a sentence.
    "T_fixedb": {
        "csv": "T20_fixed_b_decomposition.csv",
        "cols": ["reading", "bandwidth", "critical_value", "size_pct"],
        "head": ["Reading of the cluster-mean series", "$L$",
                 "Critical value", "Size"],
        "fmt": {"bandwidth": "int", "critical_value": "2", "size_pct": "1"},
        "align": "lrrr",
        "label": "tab:fixedb",
        "rename": {"reading": {
            "t_plain": "One-sample $t$, no HAC correction",
            "hac_auto_t": "HAC, automatic bandwidth, against $t$",
            "hac_wide_t": "HAC, fixed-$b$ bandwidth, against $t$",
            "hac_wide_fixedb": "HAC, fixed-$b$ bandwidth, against fixed-$b$",
        }},
    },
    # The equal-mean, non-symmetric processes. Their own properties decide how
    # the size table is read, so they are tabulated beside it.
    "T_squared": {
        "csv": "T10_inference_size.csv",
        "where_in": {"comparison": ["squared_loss", "squared_skewed",
                                    "squared_mixture",
                                    "squared_skewed_serial"]},
        "cols": ["comparison", "weekly_hac4", "cluster_cr1", "wild_bootstrap",
                 "hac_quarterly", "moving_block", "ibragimov_muller_q4",
                 "hac_fixed_b025", "quarterly_sign"],
        "head": ["Process", "HAC(4)", "CR1", "Wild", "HAC-$q$", "Block",
                 "IM", "Fx-$b$", "Sign"],
        "fmt": {"weekly_hac4": "pct1", "cluster_cr1": "pct1",
                "wild_bootstrap": "pct1", "hac_quarterly": "pct1",
                "moving_block": "pct1", "ibragimov_muller_q4": "pct1",
                "hac_fixed_b025": "pct1", "quarterly_sign": "pct1"},
        "rename": {"comparison": {
            "squared_loss": "symmetric ratio",
            "squared_skewed": "skewed ratio",
            "squared_mixture": "contaminated ratio",
            "squared_skewed_serial": "skewed ratio, persistent"}},
        "align": "lrrrrrrrr",
        "label": "tab:squared",
        "small": True,
    },
    "T_size_stress": {
        "csv": "T10_inference_size.csv",
        "where_in": {"comparison": ["serial_strong", "long_memory", "regime",
                                    "few_clusters"]},
        "cols": ["comparison", "weekly_hac4", "cluster_cr1", "wild_bootstrap",
                 "quarterly_sign", "hac_quarterly", "moving_block",
                 "ibragimov_muller_q2", "ibragimov_muller_q4",
                 "ibragimov_muller_q8", "hac_fixed_b025"],
        "head": ["Process", "HAC(4)", "CR1", "Wild", "Sign", "HAC-$q$",
                 "Block", "IM $q{=}2$", "IM $q{=}4$", "IM $q{=}8$",
                 "Fx-$b$"],
        "fmt": {"weekly_hac4": "pct1", "cluster_cr1": "pct1",
                "wild_bootstrap": "pct1", "quarterly_sign": "pct1",
                "hac_quarterly": "pct1", "moving_block": "pct1",
                "ibragimov_muller_q2": "pct1", "ibragimov_muller_q4": "pct1",
                "ibragimov_muller_q8": "pct1", "hac_fixed_b025": "pct1"},
        "align": "lrrrrrrrrrr",
        "label": "tab:sizestress",
        "small": True,
        # Eleven columns overflow the text block by 8pt without this.
        "resize": True,
    },
    "T_size_real": {
        "csv": "T10_inference_size.csv",
        "where": {"part": "B_randomization_on_real_data"},
        "cols": ["comparison", "dates_per_quarter", "excess_kurtosis",
                 "weekly_hac4", "cluster_cr1", "quarterly_t",
                 "wild_bootstrap", "quarterly_sign", "ibragimov_muller_q4",
                 "hac_fixed_b025"],
        "head": ["Comparison", "$m$", "Kurt.", "HAC(4)", "CR1",
                 "Quart.\\ $t$", "Wild", "Sign", "IM", "Fx-$b$"],
        "fmt": {"dates_per_quarter": "int", "excess_kurtosis": "0",
                "weekly_hac4": "pct1", "cluster_cr1": "pct1",
                "quarterly_t": "pct2", "wild_bootstrap": "pct1",
                "quarterly_sign": "pct1", "ibragimov_muller_q4": "pct1",
                "hac_fixed_b025": "pct1"},
        "align": "lrrrrrrrrr",
        "label": "tab:sizereal",
        "resize": True,
        "small": True,
    },
    "T_multiplicity": {
        "csv": "T15_multiplicity.csv",
        "cols": ["model", "benchmark", "estimand", "dependence", "p_raw",
                 "p_holm", "p_bh", "status"],
        "head": ["Model", "Benchmark", "Estimand", "Dependence", "$p$",
                 "$p$ Holm", "$p$ BH", "Inferential status"],
        "fmt": {"p_raw": "4", "p_holm": "4", "p_bh": "4"},
        "rename": {"status": {
            "significant after Holm; conditional on independence":
                "sig.\\ (Holm) --- CONDITIONAL",
            "significant after BH only; conditional on independence":
                "sig.\\ (BH only) --- CONDITIONAL",
            "significant unadjusted only; conditional on independence":
                "sig.\\ unadj.\\ only --- CONDITIONAL",
            "significant after Holm": "significant (Holm)",
            "significant after BH only": "significant (BH only)",
            "significant unadjusted only": "significant unadjusted only",
            "not significant": "not significant",
        }},
        "align": "llllrrrl",
        "sort": "p_raw",
        "max": 14,
        "label": "tab:multiplicity",
        "resize": True,
        "small": True,
    },
    "T_blocks": {
        "csv": "T14_block_robustness.csv",
        "cols": ["model", "benchmark", "rho1_quarterly", "p_runs_on_signs",
                 "p_wild_block1", "p_wild_block2", "p_wild_block4",
                 "p_wild_block8", "p_sign_block1", "p_sign_block2",
                 "p_sign_block4", "p_sign_block8"],
        "head": ["Model", "Benchmark", "$\\rho_1$", "Runs $p$",
                 "W $b{=}1$", "W $b{=}2$", "W $b{=}4$", "W $b{=}8$",
                 "S $b{=}1$", "S $b{=}2$", "S $b{=}4$", "S $b{=}8$"],
        "fmt": {"rho1_quarterly": "2", "p_runs_on_signs": "3",
                "p_wild_block1": "3", "p_wild_block2": "3",
                "p_wild_block4": "3", "p_wild_block8": "3",
                "p_sign_block1": "3", "p_sign_block2": "3",
                "p_sign_block4": "3", "p_sign_block8": "3"},
        "align": "llrrrrrrrrrr",
        "label": "tab:blocks",
        "resize": True,
        "small": True,
    },
    # Fifteen columns at scriptsize scaled to the text width is not readable
    # on paper. The table splits by what a column ASSUMES: panel A the
    # procedures that treat target quarters as independent, panel B those that
    # admit dependence between them, which is also the study's argument
    #.
    "T_procedures_indep": {
        "csv": "T13_procedure_audit.csv",
        "cols": ["model", "benchmark", "n_clusters", "p_weekly_hac",
                 "p_cluster_cr1", "p_cluster_cr2", "p_cluster_cr3",
                 "p_wild_bootstrap", "p_quarterly_t", "p_wilcoxon", "p_sign"],
        "head": ["Model", "Benchmark", "$G$", "HAC(4)", "CR1", "CR2", "CR3",
                 "Wild", "Quart.\\ $t$", "Wilc.", "Sign"],
        "fmt": {"n_clusters": "int", "p_weekly_hac": "3",
                "p_cluster_cr1": "3", "p_cluster_cr2": "3",
                "p_cluster_cr3": "3", "p_wild_bootstrap": "3",
                "p_quarterly_t": "3", "p_wilcoxon": "3", "p_sign": "3"},
        "align": "llrrrrrrrrr",
        "label": "tab:proceduresindep",
        "resize": True,
        "small": True,
    },
    "T_procedures_dep": {
        "csv": "T13_procedure_audit.csv",
        "cols": ["model", "benchmark", "n_clusters", "p_hac_quarterly",
                 "p_moving_block", "p_ibragimov_muller", "p_hac_fixed_b"],
        "head": ["Model", "Benchmark", "$G$", "HAC-$q$", "Block", "IM",
                 "Fixed-$b$"],
        "fmt": {"n_clusters": "int", "p_hac_quarterly": "3",
                "p_moving_block": "3", "p_ibragimov_muller": "3",
                "p_hac_fixed_b": "3"},
        "align": "llrrrrr",
        "label": "tab:proceduresdep",
        "small": True,
    },
    "T_power_adj": {
        "csv": "T19_power.csv",
        "cols": ["process", "mu", "wild_bootstrap_adj", "cluster_cr1_adj",
                 "moving_block_adj", "quarterly_sign_adj",
                 "hac_fixed_b025_adj", "ibragimov_muller_q4_adj"],
        "head": ["Process", "$\\mu$", "Wild", "CR1", "Block", "Sign",
                 "Fixed-$b$", "IM ($q{=}4$)"],
        "fmt": {"mu": "2", "wild_bootstrap_adj": "3", "cluster_cr1_adj": "3",
                "moving_block_adj": "3", "quarterly_sign_adj": "3",
                "hac_fixed_b025_adj": "3", "ibragimov_muller_q4_adj": "3"},
        "rename": {"process": {"serial": "serial",
                               "heavy_tail": "heavy tail"}},
        "align": "llrrrrrr",
        "label": "tab:power",
        "small": True,
    },
    # The single-factor experiment across families. One declared hypothesis
    # evidenced three times, not three hypotheses, so it does not enter the
    # multiplicity families.
    # Coverage differs sharply across arms, so a ranking by pairwise RMSE
    # compares numbers computed on different states of the world. Three
    # readings, side by side (the design notes).
    # The self-contained calibration for the study, so its reader does
    # not depend on an unpublished companion for the decision every verdict
    # rests on (the design notes).
    "T_calib": {
        "csv": "T27_calibration_this_design.csv",
        "cols": ["process", "weekly_dm", "wild_cluster", "fixed_b", "im_q4"],
        "head": ["Process", "Weekly DM", "Wild cluster", "Fixed-$b$",
                 "IM ($q{=}4$)"],
        "fmt": {"weekly_dm": "1", "wild_cluster": "1", "fixed_b": "1",
                "im_q4": "1"},
        "align": "lrrrr",
        "label": "tab:calib",
        "small": True,
    },
    "T_balanced": {
        "csv": "T26_balanced_benchmark.csv",
        "cols": ["model", "coverage", "rel_pairwise", "rel_balanced",
                 "rel_operational"],
        "head": ["Model", "Coverage", "Pairwise", "Balanced", "Operational"],
        "fmt": {"coverage": "2", "rel_pairwise": "3", "rel_balanced": "3",
                "rel_operational": "3"},
        "align": "lrrrr",
        "label": "tab:balanced",
        "resize": True,
        "small": True,
    },
    "T_singlefactor": {
        "csv": "T21_single_factor_families.csv",
        "cols": ["family", "rmse_completed", "rel_rmse",
                 "rel_rmse_ex_pandemic", "quarters_completed_ahead",
                 "p_weekly_hac", "p_ibragimov_muller", "p_hac_fixed_b",
                 "p_wild_bootstrap"],
        "head": ["Completer", "RMSE", "Rel.\\ RMSE", "Ex-pandemic",
                 "$q$ ahead", "Weekly", "IM", "Fixed-$b$", "Wild"],
        "fmt": {"rmse_completed": "3", "rel_rmse": "3",
                "rel_rmse_ex_pandemic": "3",
                "quarters_completed_ahead": "int", "p_weekly_hac": "3",
                "p_ibragimov_muller": "3", "p_hac_fixed_b": "3",
                "p_wild_bootstrap": "3"},
        "rename": {"family": {"chronos": "Chronos-Bolt",
                              "timesfm": "TimesFM 2.5",
                              "lagllama": "Lag-Llama"}},
        "align": "lrrrrrrrr",
        "label": "tab:singlefactor",
        "small": True,
    },
    # Is the leverage result a property of the design, or of squared error and
    # one pandemic quarter? Both objections are answerable (the design notes).
    "T_robustness": {
        "csv": "T23_robustness_summary.csv",
        "cols": ["loss", "sample", "ess_lo", "ess_hi", "nu_lo", "nu_hi",
                 "tmax_lo", "tmax_hi", "empty_region"],
        "head": ["Loss", "Sample", "$G_{\\mathrm{eff}}$ min",
                 "max", "$\\nu$ min", "max", "$t_{\\max}$ min", "max",
                 "No region"],
        "fmt": {"ess_lo": "1", "ess_hi": "1", "nu_lo": "2", "nu_hi": "2",
                "tmax_lo": "2", "tmax_hi": "2", "empty_region": "int"},
        "align": "llrrrrrrr",
        "label": "tab:robustness",
        "small": True,
    },
    # One application table, carrying effect size beside the procedures that
    # decide. The bandwidth sweep and the cluster-robust variants move to the
    # supplement: they are the audit trail, not the result (the design notes).
    "T_app": {
        "csv": "T24_application.csv",
        "cols": ["model", "benchmark", "rel_rmse", "pct_ahead", "n_clusters",
                 "ess_serial", "p_weekly_hac", "p_wild_bootstrap", "p_sign",
                 "p_ibragimov_muller", "p_hac_fixed_b"],
        "head": ["Model", "Benchmark", "Rel.\\ RMSE", "\\% ahead", "$G$",
                 "$G_{\\mathrm{eff}}$", "Weekly", "Wild", "Sign", "IM",
                 "Fx-$b$"],
        "fmt": {"rel_rmse": "3", "pct_ahead": "0", "n_clusters": "int",
                "ess_serial": "1", "p_weekly_hac": "3",
                "p_wild_bootstrap": "4", "p_sign": "3",
                "p_ibragimov_muller": "3", "p_hac_fixed_b": "3"},
        "align": "llrrrrrrrrr",
        "label": "tab:application",
        "resize": True,
        "small": True,
    },
    "T_ess": {
        "csv": "T16_effective_sample.csv",
        "cols": ["model", "benchmark", "n_pairs", "n_clusters", "rho1",
                 "skew_quarterly", "ess_serial", "satterthwaite_df",
                 "t_max_randomization", "t_critical_conventional"],
        "head": ["Model", "Benchmark", "Pairs $n$", "Clusters $G$",
                 "$\\rho_1$", "$\\gamma_1$", "$G_{\\mathrm{eff}}$", "$\\nu$",
                 "$t_{\\max}$", "$t_{0.975}(G{-}1)$"],
        "fmt": {"n_pairs": "int", "n_clusters": "int", "rho1": "2",
                "skew_quarterly": "1",
                "ess_serial": "1", "satterthwaite_df": "2",
                "t_max_randomization": "2", "t_critical_conventional": "2"},
        "align": "llrrrrrrrr",
        "label": "tab:ess",
        "resize": True,
        "small": True,
    },
    "T_audit": {
        "csv": "T8_inference_audit.csv",
        "cols": ["model", "benchmark", "n_quarters", "rmse_model",
                 "rmse_benchmark", "p_hac4", "p_hac17", "p_t_quarterly",
                 "p_sign", "p_wilcoxon"],
        "head": ["Model", "Benchmark", "$n_q$", "RMSE", "RMSE bench.",
                 "$p$ HAC(4)", "$p$ HAC(17)", "$p$ quart.\\ $t$",
                 "$p$ sign", "$p$ Wilcox."],
        "fmt": {"n_quarters": "int", "rmse_model": "3", "rmse_benchmark": "3",
                "p_hac4": "3", "p_hac17": "3", "p_t_quarterly": "3",
                "p_sign": "4", "p_wilcoxon": "4"},
        "align": "llrrrrrrrr",
        "label": "tab:audit",
        "resize": True,
        "small": True,
    },
    "T_naive": {
        "csv": "T7_vs_naive_benchmark.csv",
        # The verdict columns this table used to carry ("lower mean loss",
        # "indistinguishable") were read off the wild cluster bootstrap, which
        # Table~\\ref{tab:calib} puts at 20.4% size under the persistence we
        # measure. The table now leads with the Ibragimov-Muller test the
        # paper declares primary and keeps the bootstrap and the sign test
        # beside it, so no column asserts a verdict the text declines to make.
        "cols": ["model", "n_quarters", "rmse_model", "rmse_rel_naive",
                 "quarters_ahead", "p_im_q4", "p_wild_bootstrap", "p_sign"],
        "head": ["Model", "$n_q$", "RMSE", "Rel.\\ RMSE vs mean", "$q$ ahead",
                 "$p$ IM ($q{=}4$)", "$p$ wild", "$p$ sign"],
        "fmt": {"n_quarters": "int", "rmse_model": "3",
                "rmse_rel_naive": "3", "quarters_ahead": "int",
                "p_im_q4": "3", "p_wild_bootstrap": "3", "p_sign": "4"},
        "align": "lrrrrrrr",
        "sort": "rmse_rel_naive",
        "label": "tab:naive",
        "resize": True,
        "small": True,
    },
    "T_loss": {
        "csv": "T28_loss_and_concentration.csv",
        "cols": ["model", "n_quarters", "rel_rmse", "rel_mae", "rel_med_ae",
                 "loo_rel_rmse_min", "loo_rel_rmse_max", "share_abs_top1",
                 "share_abs_top5"],
        "head": ["Model", "$n_q$", "Rel.\\ RMSE", "Rel.\\ MAE",
                 "Rel.\\ med.\\ AE", "LOO min", "LOO max",
                 "Top 1", "Top 5"],
        "fmt": {"n_quarters": "int", "rel_rmse": "3", "rel_mae": "3",
                "rel_med_ae": "3", "loo_rel_rmse_min": "3",
                "loo_rel_rmse_max": "3", "share_abs_top1": "2",
                "share_abs_top5": "2"},
        "align": "lrrrrrrrr",
        "sort": "rel_rmse",
        "label": "tab:loss",
        "resize": True,
        "small": True,
    },
    "T_imputation": {
        "csv": "T32_imputation_both.csv",
        "cols": ["arm", "quantile", "rel_rect", "rel_cov"],
        "head": ["Completion", "Quantile", "Rectangular context",
                 "With covariates"],
        "fmt": {"quantile": "2", "rel_rect": "3", "rel_cov": "3"},
        "align": "lrrr",
        "label": "tab:imputation",
        "small": True,
    },
    "T_interface": {
        "csv": "T31_completed_arms.csv",
        "cols": ["model", "n_quarters", "rel_rmse_kalman", "quarters_ahead",
                 "p_im_q4", "rel_rmse_naive", "p_im_q4_naive"],
        "head": ["Completed by", "$n_q$", "Rel.\\ RMSE vs Kalman",
                 "$q$ ahead", "$p$ IM ($q{=}4$)", "Rel.\\ RMSE vs mean",
                 "$p$ IM ($q{=}4$)"],
        "fmt": {"n_quarters": "int", "rel_rmse_kalman": "3",
                "quarters_ahead": "int", "p_im_q4": "3",
                "rel_rmse_naive": "3", "p_im_q4_naive": "3"},
        "align": "lrrrrrr",
        "label": "tab:interface",
        "resize": True,
        "small": True,
    },
    "T_stacking": {
        "csv": "T33_stacking_order.csv",
        "cols": ["model", "benchmark", "n", "p_target_first", "p_calendar",
                 "p_horizon_first"],
        "head": ["Model", "Benchmark", "$n$", "Target first", "Calendar",
                 "Horizon first"],
        "fmt": {"n": "int", "p_target_first": "4", "p_calendar": "4",
                "p_horizon_first": "4"},
        "align": "llrrrr",
        "label": "tab:stacking",
        "resize": True,
        "small": True,
    },
    "T_panel": {
        "csv": "T34_indicator_panel.csv",
        "cols": ["series", "description", "frequency", "transformation",
                 "vintages_from"],
        "head": ["Series", "Description", "Frequency", "Transformation",
                 "Vintages from"],
        "fmt": {},
        "align": "lllll",
        "label": "tab:panel",
        "resize": True,
        "small": True,
    },
    "T_cross": {
        "csv": "T9_cross_economy.csv",
        "cols": ["region", "sample", "model", "benchmark", "n_quarters",
                 "rmse_model", "rmse_benchmark", "rmse_ratio",
                 "quarters_model_ahead", "p_t", "p_sign"],
        "head": ["Region", "Sample", "Model", "Benchmark", "$n_q$", "RMSE",
                 "RMSE bench.", "Ratio", "$q$ ahead", "$p$ $t$", "$p$ sign"],
        "fmt": {"n_quarters": "int", "rmse_model": "2", "rmse_benchmark": "2",
                "rmse_ratio": "3", "quarters_model_ahead": "int",
                "p_t": "3", "p_sign": "3"},
        "align": "llllrrrrrrr",
        "label": "tab:cross",
        "resize": True,
        "small": True,
    },
    "T_cycle": {
        "csv": "F5_within_quarter_cycle.csv",
        "cols": ["model", "rmse_first_week", "rmse_best_week", "best_week",
                 "rmse_last_week", "gain_to_best_pct", "gain_at_end_pct"],
        "head": ["Model", "RMSE wk 13", "RMSE best", "Best wk",
                 "RMSE wk $-3$", "Gain to best (\\%)", "Gain at end (\\%)"],
        "fmt": {"rmse_first_week": "2", "rmse_best_week": "2",
                "best_week": "int", "rmse_last_week": "2",
                "gain_to_best_pct": "1", "gain_at_end_pct": "1"},
        "align": "lrrrrrr",
        "sort": "-gain_to_best_pct",
        "label": "tab:cycle",
        "resize": True,
        "small": True,
    },
    "T_power": {
        "csv": "T11_power.csv",
        "cols": ["comparison", "n_quarters", "quarters_model_ahead", "p_hat",
                 "power_at_observed_n", "n_quarters_for_80pct_power",
                 "years_for_80pct_power", "years_optimistic"],
        "head": ["Comparison", "$n_q$", "$q$ ahead", "$\\hat p$",
                 "Power here", "$q$ for 80\\%", "Years", "Years (opt.)"],
        "fmt": {"n_quarters": "int", "quarters_model_ahead": "int",
                "p_hat": "3", "power_at_observed_n": "3",
                "n_quarters_for_80pct_power": "int",
                "years_for_80pct_power": "int", "years_optimistic": "int"},
        "align": "lrrrrrrr",
        "where": {"kind": "observed_comparison"},
        "label": "tab:power",
        "resize": True,
        "small": True,
    },
    "T_powergrid": {
        "csv": "T11_power.csv",
        "cols": ["p_hat", "effect_size_from_half",
                 "n_quarters_for_80pct_power", "years_for_80pct_power",
                 "power_at_82_quarters"],
        "head": ["$p$", "$p-0.5$", "$q$ for 80\\%", "Years",
                 "Power at 82 $q$"],
        "fmt": {"p_hat": "3", "effect_size_from_half": "3",
                "n_quarters_for_80pct_power": "int",
                "years_for_80pct_power": "1", "power_at_82_quarters": "3"},
        "align": "rrrrr",
        "where": {"kind": "effect_size_grid"},
        "label": "tab:powergrid",
    },
    "T_variance": {
        "csv": "T6_variance_decomposition.csv",
        "cols": ["spec", "term", "n_params", "r2"],
        "head": ["Specification", "Terms", "Parameters", "$R^2$"],
        "fmt": {"n_params": "int", "r2": "4"},
        "align": "llrr",
        "label": "tab:variance",
    },
    "T_finetune": {
        "csv": "T12_finetune_budget.csv",
        "cols": ["mode", "finetune_steps", "n_quarters", "rmse",
                 "deviation_pct", "p_t_quarterly", "p_sign"],
        "head": ["Route", "Steps", "$n_q$", "RMSE", "Deviation (\\%)",
                 "$p$ quart.\\ $t$", "$p$ sign"],
        "fmt": {"finetune_steps": "int", "n_quarters": "int", "rmse": "4",
                "deviation_pct": "3", "p_t_quarterly": "3", "p_sign": "3"},
        "align": "lrrrrrr",
        "label": "tab:finetune",
        "resize": True,
    },
    "T_ablation": {
        "csv": "T5_ragged_edge_ablation.csv",
        "cols": ["model", "rmse_real_ragged", "rmse_balanced_expost",
                 "rmse_reduction_pct", "n"],
        "head": ["Model", "RMSE, real ragged edge",
                 "RMSE, ex-post balanced", "Reduction (\\%)", "$n$"],
        "fmt": {"rmse_real_ragged": "3", "rmse_balanced_expost": "3",
                "rmse_reduction_pct": "1", "n": "int"},
        "align": "lrrrr",
        "label": "tab:ablation",
        "resize": True,
    },
    "T_mcs": {
        "csv": "T2_model_confidence_set.csv",
        "cols": None,
        "head": None,
        "fmt": {},
        "align": None,
        "label": "tab:mcs",
        "resize": True,
        "small": True,
    },
}
# fmt: on


def _cell(value, spec: str | None) -> str:
    """Format one cell for LaTeX text mode.

    Underscores, ampersands and percent signs are escaped unconditionally:
    in text mode an unescaped underscore is a hard compile error, not a
    stylistic choice, and model identifiers here are full of them.
    """
    if pd.isna(value):
        return "---"
    if isinstance(value, str):
        return (
            value.replace("\\", "")
            .replace("&", "\\&")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
    if spec == "int":
        return f"{int(round(float(value)))}"
    if spec == "trim":
        # One decimal, but only where there is one: the unbalanced design
        # averages 9.5 dates per quarter and printing it as 9 contradicted
        # the parameter table beside it.
        v = round(float(value), 1)
        return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"
    if spec and spec.startswith("pct"):
        digits = int(spec[3:]) if len(spec) > 3 else 1
        return f"{100 * float(value):.{digits}f}"
    digits = int(spec) if spec else 3
    return f"{float(value):.{digits}f}"


# Readable names for the model identifiers. The tables used to print the
# code identifiers, which a reader of the study cannot decode
# because that paper describes no models.
from src.labels import MODEL_NAMES, PROCESS_NAMES

# Every option build_one understands. A spec key outside this set is a
# silently ignored instruction, which is how tab:multiplicity ended up with a
# caption describing a table it was not producing.
_SPEC_KEYS = frozenset(
    {
        "csv",
        "where",
        "where_in",
        "cols",
        "sort",
        "max",
        "rename",
        "head",
        "align",
        "fmt",
        "label",
        "resize",
        "small",
    }
)


def build_one(name: str, spec: dict) -> str:
    unknown = sorted(set(spec) - _SPEC_KEYS)
    if unknown:
        raise KeyError(
            f"table spec {name!r} names options build_one does not implement: "
            f"{unknown}. Either implement them or remove them -- an ignored "
            f"option produces a table that silently differs from its caption."
        )
    df = pd.read_csv(TABLES / spec["csv"])
    for col, val in (spec.get("where") or {}).items():
        df = df[df[col].astype(str) == val]
    for col, vals in (spec.get("where_in") or {}).items():
        df = df[df[col].astype(str).isin([str(v) for v in vals])]
    if spec.get("cols"):
        # Loudly, not silently. This filter used to drop requested columns
        # that were absent, so renaming `hac_fixed_b` to `hac_fixed_b025` in
        # the size study removed that column from the study's own size table
        # with no error and no visible gap.
        absent = [c for c in spec["cols"] if c not in df.columns]
        if absent:
            raise KeyError(
                f"table spec {name!r} requests columns absent from "
                f"{spec['csv']}: {absent}. Available: {list(df.columns)}"
            )
        df = df[list(spec["cols"])]
    if spec.get("sort"):
        col = spec["sort"].lstrip("-")
        if col in df.columns:
            df = df.sort_values(col, ascending=not spec["sort"].startswith("-"))
    # `max` and `rename` were in the specs long before they were implemented
    # here, and were silently ignored: tab:multiplicity's caption promised the
    # twelve smallest p-values while the fragment carried all forty-six
    #. _SPEC_KEYS below now fails the build if a
    # spec names an option this function does not consume.
    if spec.get("max"):
        df = df.head(int(spec["max"]))
    for col, mapping in (spec.get("rename") or {}).items():
        if col in df.columns:
            df[col] = df[col].map(lambda v, m=mapping: m.get(v, v))
    # Model and process identifiers are translated everywhere they appear,
    # without each spec having to ask.
    for col in ("model", "benchmark"):
        if col in df.columns:
            df[col] = df[col].map(lambda v: MODEL_NAMES.get(v, v))
    if "comparison" in df.columns:

        def _pretty(v: str) -> str:
            if " vs " in str(v):
                a_, b_ = str(v).split(" vs ", 1)
                return f"{MODEL_NAMES.get(a_, a_)} vs " f"{MODEL_NAMES.get(b_, b_)}"
            return PROCESS_NAMES.get(v, v)

        df["comparison"] = df["comparison"].map(_pretty)
    if "label" in df.columns:
        df["label"] = df["label"].map(
            lambda v: re.sub(
                r"\(([a-z_]+)\)",
                lambda m: f"({PROCESS_NAMES.get(m.group(1), m.group(1))})",
                str(v),
            )
        )
    head = spec.get("head") or [str(c).replace("_", " ") for c in df.columns]
    align = spec.get("align") or "l" + "r" * (len(df.columns) - 1)

    lines = []
    if spec.get("small"):
        lines.append("\\footnotesize")
    # A ten- or eleven-column table does not fit elsarticle's single-column
    # text width at 12pt, and an overfull box of 200pt is text running off
    # the page rather than a cosmetic warning. \resizebox scales the whole
    # tabular to the text width, which is the standard remedy and keeps the
    # column set intact -- dropping columns to fit would hide the audit
    # trail this study is about.
    if spec.get("resize"):
        lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(head) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        cells = [_cell(v, spec["fmt"].get(c)) for c, v in zip(df.columns, row)]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if spec.get("resize"):
        lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written, skipped = [], []
    for name, spec in SPECS.items():
        if not (TABLES / spec["csv"]).exists():
            skipped.append(f"{name} ({spec['csv']} missing)")
            continue
        (out / f"{name}.tex").write_text(build_one(name, spec))
        written.append(name)
    print(f"wrote {len(written)} table fragments to {out}: {', '.join(written)}")
    if skipped:
        print("skipped: " + "; ".join(skipped))


if __name__ == "__main__":
    main()
