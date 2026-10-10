# Data

The raw series are exported from a Bloomberg Terminal and are licensed, so they are **not**
tracked in this repository (`data/*` is git-ignored). To reproduce the results, export the
following and save them in this folder with exactly these names.

| File | Bloomberg ticker | Fields | Notes |
|---|---|---|---|
| `EURUSD_OHLC_2021-2026_daily.xlsx` | `EURUSD BGN Curncy` (BGN source) | `PX_OPEN`, `PX_HIGH`, `PX_LOW`, `PX_LAST` | daily, 2021-09-23 to 2026-09-23; 6 header rows before the table |
| `EURUSD_HVT_2021-2026_daily.xlsx` | `EURUSD Curncy` | `VOLATILITY_10D`, `VOLATILITY_30D`, `VOLATILITY_90D` | `=BDH(...)` export, two-row header |
| `EURUSD_implied_vol_2021-2026.xlsx` | `EURUSDV1M Curncy` | `PX_LAST` | `=BDH(...)` export, no header row |

`notebooks/01_data_preparation.ipynb` turns these into the files below (also git-ignored):

- `eurusd_model_ready.csv` - features and the 22-day-ahead target (1,240 rows)
- `eurusd_returns_full.csv` - daily log returns on the full sample, used to fit GARCH
- `windows_N64.npz` - standardised 64-day windows for TimeMixer
