# Phase 2 Preregistration — Universe Audit, Dollar-Neutral Baseline, Decision Rule

Locked before computing any Phase 2 result. The Phase 1 pack
(`repro_pack/`, commit `c59d022`) is frozen evidence: test window
January 2020 – December 2024, original negative conclusion unchanged.
Nothing in Phase 2 alters the Phase 1 test period, decision rule, or
conclusion.

## 1. Dollar-neutral baseline (defined before looking at its result)

**Baseline: short-term reversal (1-1).** At each month-end `t`, rank
active S&P 500 constituents on the prior one-month return (`t-1` to
`t`, i.e. the return skipped by the 12-1 formation), hold the bottom
decile long and the top decile short, equal-weight, dollar-neutral,
monthly rebalance, minimum 20 stocks per leg, 10 bps one-way costs with
the same drift-adjusted turnover accounting as the momentum strategy.

**Why this comparator.** It uses identical machinery (same universe,
same decile/weight/cost construction, same test window), is
dollar-neutral like the strategy under test, and is the canonical
opposite-signed signal (Jegadeesh 1990; Lehmann 1990): the 12-1 rule
explicitly skips month `t` to avoid short-term reversal contamination,
so the skipped month is the natural preregistered control. No
directional prediction is made; the baseline is contextual and does not
enter the decision rule.

**What is fixed.** Construction parameters above, test window
2020–2024, net-of-cost comparison on AnnRet / Sharpe / MaxDD with
Newey-West (6-lag) inference. No alternative baseline will be
substituted after observing the outcome.

## 2. Decision rule (primary claim separated from secondary evidence)

- **Primary claim (sign of premium).** The 12-1 strategy earns a
  positive mean net premium on the locked test window. Decided solely
  by: (a) mean net monthly return > 0, (b) net Sharpe (2% rf) > 0,
  (c) HAC(6) two-sided p-value for mean excess return < 0.05.
- **Secondary evidence (factor alpha).** FF5+UMD annualized alpha on the
  test window is reported for interpretation only. It can neither rescue
  a rejected primary claim nor overturn an accepted one; a premium fully
  spanned by UMD with no alpha is still a premium, and a positive alpha
  alongside a negative premium is still a rejection.
- **Context (not decisive).** Reversal-baseline comparison, cost sweep,
  raw-return sensitivity from Phase 1. None can move the primary
  verdict.

## 3. Universe audit plan (pre-committed checks)

1. Count band: monthly membership vs the 500-company / ~503-stock
   benchmark; report months outside, mean excess, trend.
2. Conservation: per-month adds vs drops; cumulative drift.
3. Independent anchor: live re-pull of the current constituents table;
   overlap of the frozen December 2024 set with the live set, with
   2025 index changes as the documented confound.
4. Stale-ticker identification: frozen names absent from the live set
   cross-checked against trailing price availability (Yahoo) to
   separate 2025 removals from rename/M&A ghosts.
5. Strategy-level impact: share of test-window momentum-leg holdings
   drawn from the stale set (gap-aware returns should filter delisted
   names; this check quantifies it).
6. Provenance: re-executability of the membership build from source
   today (changes-table parse status).

## 4. Interpretation plan

Restate the Phase 1 negative conclusion verbatim. Report whether the
universe audit changes any test-window input (it cannot change the
frozen result, only bound its uncertainty). Report the reversal
baseline alongside, with no directional claim. Stop at interpretation;
no specification search.
