# Phase 2 Decision Summary (locked test 2020-2024)

## Primary claim: sign of 12-1 net premium
- mean monthly = -0.00189 (NEGATIVE)
- Sharpe = -0.209 (FAIL)
- HAC(6) p = 0.6495 (FAIL)
- Verdict: REJECTED (unchanged from Phase 1).

## Secondary evidence: FF5+UMD alpha (interpretive only)
- See Phase 1 ff5_umd_alpha_test.csv (alpha -4.93%, p 0.12).
- Cannot overturn the primary verdict in either direction.

## Context: preregistered 1-1 reversal baseline (not decisive)
- Reversal test AnnRet = -0.0055, Sharpe = -0.136, MaxDD = -0.2358, HAC-t = -0.47, p = 0.6365, N = 60.
- Momentum test AnnRet = -0.0224, Sharpe = -0.209.

## Universe audit impact
- See universe_audit.md. Frozen test-window inputs are unchanged; the audit bounds uncertainty only.
