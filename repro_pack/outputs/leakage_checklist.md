# Look-Ahead-Bias Audit

| Check | Result | Detail |
|---|---|---|
| Burn-in period | PASS | First net return 2006-03-31, expected on or after 2006-03-31 (13-month burn-in). |
| No pre-signal trading | PASS | net valid n=226, gross valid n=226 (expected 226). |
| Turnover initialization | PASS | First turnover value 0.000000 (expected 0.0). |
| Return hard cap | PASS | Panel min -0.4000, max 0.4000. |
| Membership breadth | PASS | Names/month min 515, mean 518, max 525 over 240 months. |
| Index integrity | PASS | Monthly spacing ok=True, duplicates ok=True, last date 2024-12-31. |
| Cost accounting identity | PASS | max|net - (gross - 10bps*turnover)| = 1.55e-16. |

Notes: signal uses prices t-13 to t-1 only; portfolio weights are shifted one month before multiplication by returns; membership masking, gap-aware returns, and contemporaneous cross-sectional winsorization are applied at formation time. Factor data are used for ex-post attribution only, never in signal construction.
