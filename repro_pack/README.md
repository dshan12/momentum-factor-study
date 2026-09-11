# 12-1 Cross-Sectional Momentum in the S&P 500 — Reproducibility Pack

Self-contained pack for reviewing one bounded claim from the momentum
research project. Runs offline on frozen inputs in under two minutes.

## 1. Hypothesis and prediction horizon

**Hypothesis (H1).** At each month-end `t`, ranking active S&P 500
constituents on cumulative log-price appreciation from `t-13` to `t-1`
(12-month formation, 1-month skip) and holding the top decile long and
the bottom decile short (equal-weight, dollar-neutral) earns a positive
mean net premium over the following month, after 10 bps one-way
transaction costs.

**Prediction horizon.** One month (`t+1`). Portfolios are rebalanced
monthly. A minimum of 20 stocks per leg is required; months below that
threshold are no-trade months.

**Signal definition.** `mom = log(P_{t-1}) - log(P_{t-13})`, ranked
cross-sectionally each month-end. Weights are shifted one month before
multiplication by realized returns (`W.shift(1) * rets`), so formation
information strictly precedes the holding period.

## 2. Dataset provenance and split logic

Frozen inputs (see `DATA_SNAPSHOT.md` for hashes and row counts), all in
`../data/cleaned/`:

- **Universe:** S&P 500 month-end membership reconstructed from
  Wikipedia change logs, January 2005 – December 2024
  (`sp500_membership_monthly.csv`).
- **Prices:** Monthly adjusted closes for the union of all historical
  members, January 2004 – December 2024
  (`monthly_adjclose_union.csv` via Yahoo Finance at build time).
- **Return panel:** Membership-masked, gap-aware monthly returns capped
  at ±40% with 1st/99th percentile cross-sectional winsorization
  (`returns_survivorship.csv`).
- **Strategy series:** Gross returns, one-way turnover, and net returns
  at 10 bps (`strategy_gross_survivorship.csv`,
  `strategy_turnover_survivorship.csv`,
  `strategy_net_survivorship.csv`), March 2006 – December 2024
  (226 valid months after 13-month burn-in).
- **Attribution:** Fama-French five factors plus momentum, decimal
  monthly (`ff5_umd_monthly.csv`, from Kenneth French library at build
  time). The market baseline is `Mkt-RF + RF`.

**Split logic (time-based, no shuffling).**

| Split | Window | Months | Role |
|---|---|---|---|
| In-sample | Mar 2006 – Dec 2014 | 106 | Reference behavior |
| Validation | Jan 2015 – Dec 2019 | 60 | Freeze cost and metric choices |
| Test / out-of-sample | Jan 2020 – Dec 2024 | 60 | Single locked evaluation |
| Full | Mar 2006 – Dec 2024 | 226 | Context only |

No parameter is fit on the test window. The one-month holding period and
one-month skip imply a one-month embargo is sufficient; no additional
purging is applied.

## 3. Baseline

**Equal-risk market baseline:** the US market total return (`Mkt-RF +
RF`) over the identical month-end index. It is reported side by side
with the strategy on every split (annualized return, volatility, Sharpe
at 2% risk-free, maximum drawdown). This is deliberately simple: no
ranking, no turnover, no cost model. Any claim of a momentum premium
must clear this hurdle on both absolute and risk-adjusted terms.

## 4. Leakage / look-ahead-bias audit

Construction controls (verified in `src/analysis_survivorship_free.py`
and `src/data/`):

1. Non-members are masked to NaN before any return or signal
   computation.
2. Monthly returns require valid prices at both `t` and `t-1`.
3. The 12-1 signal uses `t-13` to `t-1` only; the most recent month is
   skipped.
4. Target weights are shifted one month before return multiplication.
5. Turnover is computed from drift-adjusted pre-trade weights.
6. Winsorization uses the contemporaneous cross-section only.
7. Factor data are used for ex-post attribution, never in formation.

`audit_leakage.py` independently checks burn-in timing, valid-month
counts, zero-initialized turnover, return caps, membership breadth,
index integrity, and the `net = gross - cost * turnover` identity. See
`outputs/leakage_checklist.md` (PASS/FAIL per check).

## 5. Metrics and robustness

**Primary metric.** Annualized Sharpe ratio of net long-short returns at
2% annual risk-free, with a Newey-West (6-lag) t-statistic and p-value
for mean monthly excess return. Sharpe is the decision metric because it
scales the premium by its realized volatility over each fixed window.

**Robustness checks (all on frozen inputs).**

- Transaction-cost sweep at 5 / 10 / 15 / 25 bps on the test window
  (`tc_sensitivity_test.csv`).
- In-sample vs. validation vs. out-of-sample decomposition
  (`strategy_by_split.csv`, `baseline_market_by_split.csv`).
- FF5+UMD attribution on the test window with Newey-West standard
  errors (`ff5_umd_alpha_test.csv`): annualized alpha, t-statistic,
  p-value, R-squared.
- Equity and drawdown plots with the out-of-sample boundary marked
  (`equity_by_split.png`, `drawdown.png`).

## 6. Rejection rule

Reject H1 if, on the locked test window (2020–2024), **any** of the
following holds: (a) mean net monthly return ≤ 0; (b) net Sharpe ≤ 0;
(c) FF5+UMD annualized alpha ≤ 0 or statistically indistinguishable
from zero at the 5% level (Newey-West). A premium that is positive only
before costs, only in-sample, or only at unrealistically low turnover
costs does not sustain H1.

For context, the full-sample reference result is a negative net premium
(approximately −2.8% annualized, Sharpe −0.23, FF5+UMD alpha
approximately −4.0% with R² ≈ 0.82), driven by short-leg snap-back
rallies in momentum-crash months. The test-window evaluation determines
whether that conclusion holds out of sample.

## 7. Reproduction

Requirements: Python 3.13+, `pip install -r requirements-repro.txt`
(pandas, numpy, matplotlib, statsmodels only). No network access or
credentials required.

```bash
cd repro_pack
pip install -r requirements-repro.txt
python run_repro.py
python audit_leakage.py
```

Expected runtime is under two minutes on a standard machine. Outputs in
`outputs/`: `strategy_by_split.csv`, `baseline_market_by_split.csv`,
`tc_sensitivity_test.csv`, `ff5_umd_alpha_test.csv`,
`equity_by_split.png`, `drawdown.png`, `verification.log`,
`leakage_checklist.csv`, `leakage_checklist.md`.

Regenerating the frozen inputs from source (Wikipedia, Yahoo Finance,
French library, ~15–20 minutes, network required) follows the main
project `README.md` and is not part of this pack.

## Limitations

Monthly S&P 500 scope only; results do not extend to small caps,
intraday horizons, or alternative weighting. Short proceeds are assumed
available without additional borrow fees beyond modeled turnover costs.
