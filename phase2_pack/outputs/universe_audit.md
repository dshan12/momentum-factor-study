# Point-in-Time Universe Audit

| metric | value |
|---|---|
| months | 240 |
| mean members/month | 518.0 |
| min/max members | 515/525 |
| months outside [498,508] | 240 |
| mean excess vs 503 | 15.0 |
| mean adds/drops per month | 1.38/1.36 |
| cumulative drift (adds-drops) | 5 |
| frozen 2024-12-31 count | 520 |
| anchor (frozen snapshot 2026-09-11 (offline)) | 503 |
| frozen-not-live (stale-or-2025-removed) | 49 |
| non-participating (zero lifetime prices) | 24 |
| mid-window delistings (ghosts) | 0 |
| legitimate thru window (likely 2025 removals) | 25 |
| union tickers fully empty | 209 |
| tradable members/month (mean/min) | 404/329 |
| live-not-frozen | 32 |
| test months with ghost in leg | 0 |
| max ghost share of a monthly leg | 0.0000 |
| ghosts with any test-window return | 0/24 |
| rebuild module status | importable |
| changes-table re-parse today | parse FAILED: RuntimeError: Could not find S&P 500 changes table on Wikipedia. Column names may have changed — inspect the page manually. |


Confound: the anchor postdates the frozen window by ~20 months; frozen-not-live names split into non-participating (zero prices, zero impact by construction), mid-window delistings, and legitimate members priced through window end (no test-window uncertainty). See stale_tickers.csv.
