"""The indicator panel and the model set, as tables, from the configuration.

A referee should be able to reproduce the study from the article rather than
only from the code. These two tables carry what the text describes in prose:
every indicator with its frequency, stationarity transformation and the date
its vintage archive begins, and every model arm with the checkpoint or
specification it runs.

Writes results/tables/T34_indicator_panel.csv and
results/tables/T35_model_specifications.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.experiments.run_backtest import load_config

# The FRED-MD stationarity codes, as McCracken and Ng define them.
TCODE = {
    1: "level",
    2: "first difference",
    3: "second difference",
    4: "log",
    5: "log difference",
    6: "second log difference",
    7: "difference of the ratio",
}
FREQ = {"M": "monthly", "Q": "quarterly", "D": "daily", "W": "weekly"}


def _description(note: str) -> str:
    """The indicator's name, without the configuration's internal commentary.

    The notes in `configs/data.yaml` carry implementation history: which FRED
    identifier replaced which, and where a choice is recorded in the decision
    log. None of that belongs in a table a reader sees, so the note is cut at
    the first bracket, dash or semicolon that introduces it.
    """
    for sep in (" (", " --", " \u2014", ";", ","):
        i = note.find(sep)
        if i > 0:
            note = note[:i]
    return note.strip()


def indicator_panel(data_cfg: dict, panel: str = "baseline") -> pd.DataFrame:
    rows = []
    for s in data_cfg["series"]:
        if s.get("panel") != panel:
            continue
        rows.append(
            {
                "series": s["fred_id"],
                "description": _description(s.get("note", "")),
                "frequency": FREQ.get(s.get("freq", ""), s.get("freq", "")),
                "transformation": TCODE.get(s.get("tcode"), str(s.get("tcode"))),
                "vintages_from": str(s.get("vintage_start", ""))[:10],
            }
        )
    return pd.DataFrame(rows).sort_values(["frequency", "series"])


def model_specifications(cfg: dict) -> pd.DataFrame:
    """One row per configured arm, with what distinguishes it."""
    from src.labels import MODEL_NAMES

    models = cfg.get("models", {})
    rows = []
    for family, entry in models.items():
        names = list(entry) if entry else []
        params = entry if isinstance(entry, dict) else {}
        for name in names:
            spec = params.get(name) or {}
            detail = ", ".join(f"{k} = {v}" for k, v in spec.items() if v is not None)
            rows.append(
                {
                    "arm": MODEL_NAMES.get(name, name),
                    "family": family,
                    "specification": detail or "as cited",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    data_cfg = load_config("configs/data.yaml")
    cfg = load_config("configs/experiment_us.yaml")
    tab = Path("results/tables")
    tab.mkdir(parents=True, exist_ok=True)

    ind = indicator_panel(data_cfg)
    ind.to_csv(tab / "T34_indicator_panel.csv", index=False)
    print(f"wrote {tab / 'T34_indicator_panel.csv'} ({len(ind)} indicators)")
    print(ind.to_string(index=False))

    mod = model_specifications(cfg)
    mod.to_csv(tab / "T35_model_specifications.csv", index=False)
    print(f"\nwrote {tab / 'T35_model_specifications.csv'} ({len(mod)} arms)")


if __name__ == "__main__":
    main()
