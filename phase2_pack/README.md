# Phase 2 Pack — Universe Audit, Dollar-Neutral Baseline, Separated Decision Rule

Separate branch (`phase2-universe-baseline`) and separate pack. The
Phase 1 pack (`repro_pack/`, commit `c59d022`) is frozen evidence: test
window January 2020 – December 2024, original negative conclusion
unchanged. Nothing here alters the Phase 1 test period, decision rule,
or conclusion.

## Contents

- `PREREG.md` — preregistration, committed before any Phase 2 result
  was computed (baseline choice, decision-rule separation, audit plan).
- `universe_audit.py` — point-in-time S&P 500 audit (counts,
  conservation, live-anchor overlap, stale separation, strategy-level
  impact, rebuild probe).
- `reversal_baseline.py` — preregistered 1-1 short-term reversal
  baseline with identical machinery to the 12-1 rule.
- `inputs_snapshot/live_constituents_20260911.csv` — live anchor
  (503 tickers, pulled 2026-09-11) so the audit reruns offline.
- `outputs/` — `universe_audit.csv/md`, `stale_tickers.csv`,
  `reversal_net.csv`, `reversal_by_split.csv`, `decision_summary.md`.
- `INTERPRETATION.md` — short interpretation; stops at reporting.

## Reproduction

```bash
git checkout phase2-universe-baseline
pip install -r ../repro_pack/requirements-repro.txt
python phase2_pack/universe_audit.py
python phase2_pack/reversal_baseline.py
```

Offline after the snapshot (the live re-pull is opt-in via
`--live-url` and is not required). Runtime under one minute.
