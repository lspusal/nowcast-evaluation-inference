# Replication package

Code and results for "How many observations does a nowcast evaluation have?
Pseudoreplication and dependence in forecast comparison".

Data sources and what is and is not redistributed: see `DATA_SOURCES.md`.

## Contents

    src/            information set, models, evaluation, figures
    scripts/        Monte Carlo studies, application and table builders
    tests/          test suite, including the anti-lookahead checks
    configs/        data panel and backtest configuration (seeds included)
    data/processed/ the second-release GDP target
    results/preds/  the forecasts of the eleven arms the paper reports
    results/tables/ the result files every table is built from
    results/figures/ the six figures of the paper

## Requirements

Python 3.11 or later.

    pip install -r requirements.txt        # or: conda env create -f environment.yml

## Reproducing the paper

Nothing below needs a download.

| Output | Command | Time |
|---|---|---|
| Monte Carlo size study (T10; paper uses `REPS=20000`) | `make inference-size` | minutes |
| Power study (T19) | `make power` | minutes |
| Fixed-b decomposition (T20) | `make fixed-b` | minutes |
| Application, robustness, stacking order, indicator panel (T22-T24, T33, T34) | `make application` | minutes |
| LaTeX table fragments, written to `paper/tables/` | `make tables` | seconds |
| Figures | `make figures` | seconds |
| Tests | `make test` | seconds |

The tables printed in the paper are the fragments `make tables` writes; no number
in a table is typed by hand. The Monte Carlo scripts are deterministic given
`--seed`.

## Re-running the backtest

The forecasts in `results/preds/` come from the real-time backtest. To
regenerate them you need the raw vintages (about 8 GB, not distributed) and the
open-weight checkpoints (downloaded from Hugging Face on first use):

    export ALFRED_API_KEY=<key>      # free, see DATA_SOURCES.md
    make data                        # hours
    make backtest
    make eval                        # writes T13 and T16

`configs/experiment_us.yaml` is the configuration that produced the forecasts.
It lists further arms than the eleven the paper reports; those arms are the
subject of a separate study and their forecasts are not included. One forecasting
arm is evaluated over sampled paths and is not deterministic: re-running it
moves individual forecasts by a median of 0.075 percentage points.

## Verifying

    pytest -q

Four deliberate lookahead leaks are injected and each must be caught. Tests
that need files outside this package are skipped.

## Licence

MIT, see LICENSE.
