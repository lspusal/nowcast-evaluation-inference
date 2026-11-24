# Data sources

## What the paper uses

All observed data are United States macroeconomic series taken from **ALFRED**
(ArchivaL Federal Reserve Economic Data), Federal Reserve Bank of St. Louis,
https://alfred.stlouisfed.org. ALFRED stores every published vintage of a
series, which is what makes the real-time design possible.

The target is real GDP (`GDPC1`, BEA), evaluated against its second release.
The 28 indicators are listed in `configs/data.yaml`; the underlying
originators are:

| Originator | Series |
|---|---|
| U.S. Bureau of Economic Analysis | GDPC1, PCEPI, DSPIC96, EXPGS, IMPGS |
| U.S. Bureau of Labor Statistics | PAYEMS, UNRATE, CPIAUCSL, PPIACO, AWHMAN, MANEMP |
| U.S. Census Bureau | HOUST, PERMIT, RSAFS, DGORDER |
| U.S. Department of Labor | ICSA |
| Board of Governors of the Federal Reserve System | INDPRO, TCU, CPF3M, BUSLOANS, M2SL, FEDFUNDS, GS10, T10YFF, DEXUSEU |
| U.S. Energy Information Administration | DCOILWTICO |
| University of Michigan | UMCSENT |
| Moody's (via FRED) | BAA10Y |
| Nasdaq OMX (via FRED) | NASDAQCOM |

The last three are third-party series that FRED redistributes under the
originator's copyright notice.

## What is and is not in this package

| Item | Included | Why |
|---|---|---|
| Code, configs, seeds | yes | MIT licence |
| Cached model predictions (`results/preds/`) | yes | every table and figure rebuilds from them without any download |
| Second-release GDP target (`data/processed/`) | yes | a single public series |
| Raw vintage cache (`data/raw/`, about 8 GB) | **no** | bulk redistribution of FRED/ALFRED content is restricted by the FRED API Terms of Use, and three series carry third-party copyright |
| Open-weight checkpoints | no | downloaded from Hugging Face on first use: `amazon/chronos-bolt-base`, `google/timesfm-2.5-200m-pytorch`, Lag-Llama |

## Obtaining the raw data

1. Create a free API key at https://fred.stlouisfed.org/docs/api/api_key.html
2. `export ALFRED_API_KEY=<key>` then `make data` (several hours, caches to
   `data/raw/`). Each request is logged with its hash in `data/raw/manifest.json`.

Vintages are immutable in ALFRED, so the download is reproducible as long as
the archive is available. Data were retrieved in September 2026.
