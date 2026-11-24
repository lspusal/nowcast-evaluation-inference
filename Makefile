.PHONY: data backtest eval tables inference-size power fixed-b application figures test

# Raw vintages from ALFRED (hours; needs ALFRED_API_KEY, see DATA_SOURCES.md).
# Only needed to re-run the backtest. Everything below `backtest` works from the
# shipped predictions in results/preds/.
data:
	python -m src.experiments.run_backtest --config configs/data.yaml --stage download --panel baseline

backtest:
	python -m src.experiments.run_backtest --config configs/experiment_us.yaml --stage all

# Writes the procedure audit and effective-sample tables (T13, T16).
eval:
	python -m src.experiments.run_backtest --config configs/experiment_us.yaml --stage evaluate
	python -m src.experiments.run_backtest --config configs/experiment_us.yaml --stage tables

# Monte Carlo (no data): size (T10), power (T19), fixed-b decomposition (T20).
# The paper uses 20,000 replications for size; pass REPS=20000.
REPS ?= 2000
inference-size:
	PYTHONPATH=. python scripts/inference_size_study.py --reps $(REPS)
power:
	PYTHONPATH=. python scripts/inference_power_study.py --reps $(REPS)
fixed-b:
	PYTHONPATH=. python scripts/fixed_b_decomposition.py --reps $(REPS)

# From the predictions: application (T24), loss/sample robustness (T22, T23),
# stacking order (T33), indicator panel (T34).
application:
	PYTHONPATH=. python scripts/application_table.py
	PYTHONPATH=. python scripts/loss_and_sample_robustness.py
	PYTHONPATH=. python scripts/stacking_order.py
	PYTHONPATH=. python scripts/specification_tables.py

# LaTeX table fragments (written to paper/tables/) and figures F2, F9-F13.
tables:
	PYTHONPATH=. python scripts/build_latex_tables.py
figures:
	PYTHONPATH=. python -c "from src.viz import figures; print(figures.build_all())"

test:
	pytest -q
