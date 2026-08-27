# PRE-REGISTRATION — post-publication pre-FOMC announcement drift

**Declared 2026-08-16 after official schedule-lineage PASS and before opening any market return
associated with this identity. One direction, one instrument, one timing window, one hypothesis.**

## Mechanism and direction

The prior is a temporary equity risk premium immediately before scheduled FOMC decisions. The
locked direction is **long US equities before the announcement**. A negative result is not
inverted. Earlier published evidence that the effect disappeared is treated as a reason to expect
a kill; 1994–2015 is not opened as a rescue sample.

## Event universe and point-in-time state

- Primary period: 2016-01-01 through 2025-12-31, entirely after the original result was public.
- The event source is the sealed 80-row annual-schedule artifact. Seventy-nine slots end in a
  regular decision and one—March 17–18, 2020—is an explicit cancellation.
- The runner must ingest all 80 rows. Joining only to statement dates is forbidden.
- If a cancellation becomes public before entry, remain flat. If it arrives after entry, flatten
  at the first executable quote and retain the realized P&L. For the sealed 2020 case, the March
  15 5:00 p.m. release predates the entry window, so the state is `CANCELLED_NO_ENTRY`.
- Unscheduled meetings, notation votes, press conferences, minutes, speeches, and SEP labels do
  not create additional trades or filters.

## Instrument, clock, and execution

- Primary instrument: SPY common shares, long only, unlevered sleeve notional.
- Entry: buy at the first executable NBBO ask at or after 2:00:00 p.m. America/New_York on the
  XNYS session immediately before the scheduled decision day, with a 30-second fill deadline.
- Exit: sell at the last executable NBBO bid at or before 1:55:00 p.m. on decision day, using
  quotes no older than one second. This exits before the 2:00 p.m. statement jump.
- If either quote is absent, crossed, locked, stale, or outside regular trading hours, the event is
  an execution failure and contributes a flat return after any actually filled leg is closed at
  the next executable quote. It is never filled from a bar close or midpoint.
- Baseline adds 0.5 bp adverse latency/slippage per side beyond observed bid/ask and statutory
  sale fees. Stress scenarios use 1.0, 2.0, and 5.0 bp per side, 1/5/30-second latency, and
  100%/75%/50% fill ratios. Partial fills retain proportional profit and loss; rejected orders
  remain flat.
- An exchange halt carries the actual position until the first executable quote. An outage before
  entry means no trade. SPY splits and distributions are applied from point-in-time corporate
  actions; an unresolved action fails the run.

## Locked matched control

For each completed event, compute the same SPY clock-window return on the four immediately prior
same-weekday non-FOMC weeks. Exclude a control if its window overlaps any scheduled FOMC slot; do
not replace it with a later or hand-selected day. The primary observation is event net return
minus the mean of available controls. Fewer than three valid controls makes the event unusable and
fails completeness rather than silently dropping it.

## Evaluation and untouched holdout

- 2016–2022 is the frozen pre-holdout panel. 2023–2025 is the untouched holdout and may be opened
  only if ingestion, lineage, execution replay, and pre-holdout code checks pass without changing
  any parameter.
- There is no model fitting. Report the 79 completed event observations and the daily flat-filled
  sleeve curve separately; zeros between events do not inflate the effective sample size.
- Report net and stressed event Sharpe annualized by `sqrt(8)`, Newey–West mean t-statistic, DSR
  against the full union experiment count at execution, maximum drawdown, expected shortfall,
  worst event, annual results, hit rate, turnover, fill failures, and cancellation behavior.
- PBO is `NOT_DEFINED_SINGLE_IDENTITY`, never zero. Parameter perturbations are diagnostics only:
  entry ±5 minutes, exit ±5 minutes, and each declared cost/latency scenario. The primary identity
  must pass; no neighboring cell can replace it.
- Build the canonical daily diversification report against all four ALPHAC sleeves with fixed 10%
  candidate weight, bottom-decile existing-book stress mask, 2,000 circular block bootstraps,
  21-session blocks, and seed 20260816. Report confidence bounds, crisis dependence, co-tail loss,
  mean-zero control, and every leave-one-calendar-year-out recomputation.
- Capacity uses observed executable SPY quote size and a conservative 1% of trailing 21-session
  median dollar ADV. Report at least $1M, $5M, and $10M notional points with fill ratio and costs.

## Kill rules

Kill if any applies:

1. Net event Sharpe < 0.40, stressed Sharpe < 0.40, Newey–West t < 2.0, or DSR < 0.95.
2. The untouched 2023–2025 holdout has non-positive mean excess return or net Sharpe < 0.40.
3. Fewer than 79 completed events are accounted for, the canceled slot is absent, or any usable
   event has fewer than three locked controls.
4. The result depends on the announcement jump, overnight exposure outside the locked window, a
   bar midpoint, a later schedule snapshot, or exclusion of an execution failure.
5. Any admission-contract beta, correlation, confidence-bound, stressed-correlation, book-delta,
   expected-shortfall, drawdown, execution-evidence, or $5M capacity gate fails.
6. The primary result fails even if a perturbation looks better. No alternate window, futures
   substitute, sign flip, pre-2016 sample, or event subset may rescue it.

If research passes but historical NBBO depth or quote provenance is incomplete, return
`DATA-ESCALATE`, not `ADD`. Alpaca is relevant only for later shadow execution; it cannot replace
the historical research feed.

```prereg
profile: pre_fomc_announcement_drift_v1
lake_dir: data/raw/pre_fomc_announcement_drift/market_nbbo
alpha_names: spy_pre_fomc_24h_excess
allocator: event_only_long_flat
instrument: SPY
direction: long
entry_clock_et: 14:00:00_previous_session
exit_clock_et: 13:55:00_decision_session
scheduled_slots: 80
completed_events: 79
cancelled_slot: 2020-03-18
preholdout_end: 2022-12-31
holdout_start: 2023-01-01
holdout_end: 2025-12-31
```

## Primary references

- Federal Reserve annual schedule releases and historical calendar: official source URLs are
  sealed in `annual_schedule_events.parquet`.
- Lucca and Moench, *The Pre-FOMC Announcement Drift*: https://doi.org/10.1111/jofi.12196
- Kurov, Wolfe, and Gilbert, *The Disappearing Pre-FOMC Announcement Drift*:
  https://doi.org/10.2139/ssrn.3134546
