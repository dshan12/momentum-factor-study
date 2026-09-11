# Data Snapshot

Frozen inputs for this reproducibility pack. All files are read from
`../data/cleaned/` and are not modified by the pack. Outputs are written
to `repro_pack/outputs/`.

| File | Rows x Cols | SHA-256 (first 16) | Date range / notes |
|---|---|---|---|
| `strategy_net_survivorship.csv` | 226 x 2 | `1c621154a2dc2683` | 2006-03-31 to 2024-12-31, net of 10 bps one-way |
| `strategy_gross_survivorship.csv` | 252 x 2 | `195b17c88d27ec25` | 2004-01-31 to 2024-12-31 head with NaN burn-in, 226 valid |
| `strategy_turnover_survivorship.csv` | 252 x 2 | `50aab518fe6110b0` | One-way turnover, first observation 0.0 |
| `ff5_umd_monthly.csv` | 754 x 8 | `d074a9fb4ce5beda` | 1963-07-31 onward, decimal monthly returns; columns Mkt-RF, SMB, HML, RMW, CMA, RF, UMD |
| `returns_survivorship.csv` | 252 x 829 | `96655f4b88e46810` | Membership-masked, capped at +/-40 pct, cross-sectionally winsorized; audit use only |
| `sp500_membership_monthly.csv` | 124316 x 3 | `e345ddb2f8d3dc6c` | S&P 500 month-end membership, 2005-01 to 2024-12; audit use only |

Full SHA-256 values are recorded in the verification log produced by
`run_repro.py`. Re-download instructions for regenerating these files from
source (Wikipedia, Yahoo Finance, Kenneth French library) are in the main
project `README.md`; this pack does not require network access.
