# Phase 2 Interpretation

## Phase 1 conclusion (restated unchanged)

On the locked test window (2020–2024), the 12-1 long-short premium is
negative: AnnRet −2.24%, Sharpe −0.21 (HAC p 0.65); FF5+UMD alpha −4.93%
(p 0.12). H1 is rejected. The signal ranks, but the premium does not
survive costs or crash-month reversals out of sample.

## 1. Universe audit

- Membership runs 515–525 names/month (mean 518) vs the ~503-stock
  benchmark: +15 mean excess, every month above band. Adds and drops
  balance (1.38 vs 1.36/month, cumulative drift +5), so this is a level
  shift, not drift — consistent with rename/M&A ghosts accumulating
  (old tickers retained alongside successors, e.g. FB alongside META).
- Of 49 frozen-not-live names: 24 never had a single price (zero
  strategy impact by construction), 0 delisted mid-window, 25 priced
  through window end (legitimate members, plausibly genuine 2025
  removals — no test-window uncertainty).
- Strategy-level impact: 0 test months with a stale name in a leg;
  0/24 stale names have any test-window return. The gap-aware return
  construction filters them before they can matter.
- Coverage limitation: 209/828 union columns are fully empty
  (download failures); tradable members average 404 of 518 (min 329,
  early 2005). Legs always filled (≥20) in every window month.
- Provenance limitation: the Wikipedia changes table no longer parses
  with the frozen code (upstream format change), so the membership
  file cannot be regenerated identically from source today. The frozen
  files are the record; this audit bounds rather than repairs them.

Net: the universe overstates membership by ~3%, but the quantified
impact on the frozen test result is zero.

## 2. Reversal baseline (preregistered, contextual)

1-1 reversal on the locked test: AnnRet −0.55%, Sharpe −0.14
(HAC p 0.64), MaxDD −23.6%. Also negative — milder than momentum
(−2.24% / −0.21) with shallower drawdowns. No directional claim was
made and none is drawn; the comparator shows a signed
dollar-neutral signal earns no premium in this window either, so
nothing about the 12-1 ranking rescues H1.

## 3. Decision

Primary claim: REJECTED (mean < 0, Sharpe < 0, p = 0.65 — unchanged).
Secondary alpha: interpretive only, consistent (negative, R² 0.79).
Phase 2 adds bounds, not a revision.
