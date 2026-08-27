# PRE-REGISTRATION — clustered insider purchases

**Declared 2026-08-15 before any SEC insider-transaction ZIP was downloaded or any return was measured. One hypothesis. No parameter sweep.**

## Economic mechanism

An isolated insider purchase can be symbolic. Independent purchases by multiple officers or directors in a short window are costlier to coordinate by chance and may reveal a shared assessment that public prices understate issuer-specific information. This is an event-information mechanism, not price momentum, funding carry, trend, macro surprise, value, quality, or short interest.

## Data and timing

- Source: official SEC Insider Transactions Data Sets, Forms 3/4/5, January 2006 onward.
- Use non-derivative **open-market purchases only**: transaction code `P`, acquired/disposed code `A`.
- Reporting persons must be an officer or director. A filing that is only a 10% owner does not qualify.
- Aggregate by issuer CIK and reporting-owner CIK. Amendments replace the corresponding original record when the SEC identifiers permit; unresolved duplicates are excluded.
- A signal exists when at least **two distinct reporting owners** make qualifying purchases in a rolling **30-calendar-day** window and aggregate reported purchase value is at least **$100,000**.
- Availability is the SEC filing date plus **two US trading sessions**. This deliberately exceeds normal dissemination latency and prevents same-filing-day execution.
- Enter at the next available open after availability and hold for exactly **63 sessions**. Overlapping clusters for the same issuer do not restart the clock.

## Portfolio

- PIT universe: US common equities in the survivorship-free Sharadar research lake, price at
  least $5, trailing 21-session ADV at least $5 million when the signal becomes available.
  **Data-coverage correction before any portfolio return was constructed:** the first draft's
  machine block pointed to the deployed `data/lake`, but its SPY history began only on 2026-06-22.
  The pre-existing `data/lake_sharadar` is the repository's declared survivorship-free research
  lake (8,436 tickers including delisted names, 1998 onward), so it is the only source consistent
  with the prose-level PIT requirement and locked 2016 start. SPY comes from the pre-existing
  adjusted ETF research lake. This correction changed no signal, threshold, timing, or cost and
  occurred after a timezone error stopped execution, before scheduling or measuring any return.
- Each qualifying issuer is equal notional within the event book. **Clarified before any price or
  return series was loaded:** the first draft said "equal risk" without naming a volatility
  estimator. Adding one later would create a hidden parameter. Equal notional is deterministic and
  introduces no lookback or cap to tune.
- Hedge aggregate market beta with SPY using a trailing 252-session beta estimated strictly before entry; clamp each issuer beta to [0, 3]. Gross exposure is normalized to 1.0 after the hedge.
- Costs: 6 bps one-way for issuer trades and 1 bp one-way for SPY. No free fills.
- Capacity: report results at 1, 5 and 10 basis points of each issuer's 21-session ADV; any position above 1% ADV is inadmissible.

## Evaluation

- **Data-quality/calibration period:** 2006–2015. It may invalidate the dataset or implementation but cannot promote the sleeve and cannot change parameters.
- **Locked OOS period:** 2016 through the latest complete SEC quarter. It is read once after the pipeline and unit tests pass.
- Record the single hypothesis in the union experiment ledger and persist the complete OOS return curve even if killed.
- Report net Sharpe, Newey-West mean t-stat, DSR against the full union trial count, PBO context,
  drawdown, skew, beta, turnover, capacity, ordinary and stressed correlation to every ALPHAC
  sleeve, and combined-book marginal Sharpe at fixed weights. **Clarified before the return pass:**
  the marginal test gives this candidate 10% and funds it pro rata from the current equal-quarter
  core, leaving each current sleeve at 22.5%. The stressed correlation is measured on the bottom
  decile of equal-quarter ALPHAC days in the common OOS window. The candidate must improve Sharpe
  in every leave-one-calendar-year-out recomputation, not just on average.

## Kill rules

Kill if any one holds:

1. OOS DSR is below 0.95 or net Sharpe is below 0.40.
2. Newey-West mean-return t-stat is below 2.0.
3. Realized absolute SPY beta exceeds 0.10.
4. Any ordinary pairwise correlation exceeds 0.35, average correlation exceeds 0.15, or stressed correlation exceeds 0.50.
5. The sleeve does not improve the fixed-weight ALPHAC book after mean-zero and leave-one-year-out controls.
6. Capacity at the 1% ADV ceiling is below $5 million or net Sharpe falls below 0.40 at
   2x modeled costs. **Clarified before the return pass:** "fails at 2x costs" was made
   machine-checkable by applying the same 0.40 minimum Sharpe required of the base-cost result.

The signal is never inverted after a failure. The 30-day cluster window, two-insider threshold, $100,000 threshold, and 63-session hold are not swept.

```prereg
profile: insider_clusters
lake_dir: data/lake_sharadar
alpha_names: insider_purchase_cluster_30d
allocator: event_beta_hedged
source: sec_form345_official
cluster_days: 30
min_distinct_insiders: 2
min_purchase_value_usd: 100000
hold_sessions: 63
filing_delay_sessions: 2
oos_start: 2016-01-01
```

## Post-result implementation correction

The preliminary pass produced a KILL but combined adjusted log returns linearly while deducting
simple-return costs and writing through the canonical simple-return curve API. Before publication,
the return leg was corrected mechanically to `expm1(adjusted_log_return)`. No signal, timing,
threshold, holding period, hedge, cost or gate changed. The preliminary metrics remain in the
machine-readable result, and the corrected implementation receives a distinct ledger identity so
the correction can only make multiple-testing deflation more conservative.
