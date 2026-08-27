# PRE-REGISTRATION — petroleum inventory scarcity

**Declared 2026-08-15 before the EIA archive was downloaded in bulk or any strategy return was
measured. One economic hypothesis, one fixed construction, no parameter sweep.**

## Mechanism

Unexpected inventory accumulation indicates near-term physical abundance; unexpected draws
indicate scarcity. The candidate trades first-release inventory changes relative to the seasonal
change knowable before each report. Its input is a government physical-supply release, not price,
trend, futures-curve slope, carry, value or macro-surprise data already used by ALPHAC.

## Point-in-time data

- Source: EIA Weekly Petroleum Status Report dated archive. Each archived report records its
  release date, data-ending date and contemporaneous Table 4 CSV. Revised history/API snapshots
  are forbidden for the signal.
- Fixed products and research proxies: commercial crude excluding SPR → USO; total motor gasoline
  → UGA. Both products must be present; otherwise the book is flat.
- OOS starts 2016-01-01. Reports from the archive start in August 2011 and are used only to warm up
  the seasonal expectation.
- A report is available at its dated EIA release. Because the standard release is after the US
  open and holiday timing varies, the strategy always enters at the next US session open.
- Every raw CSV, source URL and SHA-256 digest is retained. A row whose current minus previous
  stock does not reconcile to EIA's reported difference within 0.002 million barrels is rejected.

## Signal and portfolio

- For each product, expected inventory change is the mean of the same ISO report-week changes in
  the five preceding calendar years. Exactly five observations are required.
- Surprise is actual change minus that expectation. Scale is the sample standard deviation of the
  preceding 52 valid, already released surprises; exactly 52 are required. The trading score is
  `-surprise / scale`, because an unexpectedly large draw is bullish. Clip scores to [-3, 3].
- Allocate each valid product in proportion to its absolute score. Hedge aggregate trailing
  252-session beta to DBC, estimated strictly before entry and clamped to [-3, 3]. Normalize total
  gross exposure, including the hedge, to 1.0. Hold until the next report's next-open rebalance.
- Costs are 6 bps one-way for USO, 10 bps for UGA and 3 bps for DBC. Report 2x costs. Proxy
  capacity is measured from trailing 21-session dollar ADV at 1, 5, 10 and 100 bps participation.
- These ETFs are a feasibility harness, not the production instrument claim. Promotion requires
  the same locked signal to replicate on point-in-time CL/RB futures with real rolls and fills.

## Evaluation and kill rules

Persist the complete 2016-latest OOS curve and record the hypothesis in the union experiment
ledger. PBO is not defined for a one-configuration probe; there is no selection surface.

Kill the research candidate if any condition holds:

1. Net Sharpe < 0.40, DSR < 0.95, Newey-West t-stat < 2.0, or Sharpe at 2x costs < 0.40.
2. Realized absolute DBC beta > 0.10.
3. Average correlation to current ALPHAC sleeves > 0.15, any pair > 0.35, or any bottom-decile
   stressed correlation > 0.50.
4. A fixed 10% allocation funded pro rata from the four equal-quarter sleeves does not improve
   combined-book Sharpe, its mean-zero control, and every leave-one-calendar-year-out result.
5. Either product's standalone contribution is non-positive. The sign is not flipped after a
   failure and neither product may be dropped after results are seen.

If all statistical and diversification gates pass but ETF capacity is below $5 million at the 1%
ADV ceiling, classify the result **DATA-ESCALATE**, not ADD: acquire production futures history and
rerun the unchanged hypothesis. Otherwise the verdict is ADD_TO_SHADOW or KILL as applicable.

```prereg
profile: eia_petroleum_inventory
lake_dir: data/lake_inventory
alpha_names: eia_petroleum_inventory_scarcity
allocator: score_weighted_dbc_beta_hedged
products: USO,UGA
seasonal_years: 5
scale_weeks: 52
oos_start: 2016-01-01
```
