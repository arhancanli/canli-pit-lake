# PRE-REGISTRATION — annual risk-factor narrative stability

**Declared 2026-08-15 after the filing-only feasibility pass and before loading any security
return associated with this signal. One hypothesis identity. No direction, section, horizon, or
portfolio sweep.**

## Economic mechanism and locked direction

Material changes to an issuer's annual risk disclosures can reveal changing operating conditions,
legal exposure, or managerial concern before those changes are fully incorporated into prices.
The published *Lazy Prices* result is the prior, not Canli Capital evidence. The locked direction
is **long stable disclosures and short changed disclosures**. A failed result is never inverted.

This first identity uses only 10-K Item 1A. MD&A, 10-Qs, earnings-call transcripts, embeddings,
sentiment, topic models, and alternative lexical distances are outside this test. Trying any of
them later consumes the candidate's second and final budgeted identity under a new preregistration.

## Point-in-time data and lineage

- Filing metadata and acceptance timestamps: SEC submissions JSON and filing index records.
- Source text: immutable SEC accession archive; unamended 10-K primary documents only.
- Filing classification: the two-digit SIC displayed on that immutable accession's SEC filing
  index page. A current sector snapshot is forbidden. **Storage clarification before manifest
  construction or any return was loaded:** SEC ignored a byte-range request and returned a 24.6 MB
  complete-submission file for a single utility accession; the accession index exposes the same
  filing-time SIC in roughly 13 KB. The index is therefore the locked source, avoiding a wasteful
  download of every exhibit while preserving accession-time lineage.
  **Observed-source clarification before any return was loaded:** a small subset of immutable
  accession index pages exposes no SIC at all. Those downloads remain valid, are counted and
  disclosed as `sic_missing_at_source`, and are not endlessly retried or filled from a current
  company snapshot. A filing lacking current-accession SIC cannot enter a signal cohort; a prior
  filing may still serve as the immediate text predecessor because its SIC is not used as a
  contemporaneous control.
- Equity history: survivorship-inclusive Sharadar SEP, ACTIONS, SF1, and TICKERS snapshots already
  on disk. Join CIK to permanent ticker identity; ticker text is never the issuer key.
- Price history uses split/dividend-adjusted returns. Entry-price and liquidity filters use the
  contemporaneous unadjusted price and volume. Delisted securities remain in the universe.
  **Delisting clarification before any return was loaded:** the Sharadar lake has delisted price
  histories but no dedicated CRSP-style delisting-return field. The runner carries a halted mark,
  realizes the move into the final observed bar, then administratively force-flats at that final
  observed open with normal transaction cost; it never fills missing future returns with zero.
  Every such event is reported. A result that otherwise passes but contains any force-flat event is
  `DATA-ESCALATE` pending dedicated delisting payouts/returns, not `ADD`.
- Every raw filing and header retains source URL, retrieval time, SHA-256, CIK, accession, form,
  report date, filing date, acceptance timestamp, primary document, parser version, section hash,
  and predecessor accession.
- **Lineage-seal clarification before any return was loaded:** the completed corpus result binds
  every immutable Parquet part by ordered filename, byte length, and SHA-256. The pair result binds
  that corpus digest, the exact source-manifest SHA-256, and the pair Parquet SHA-256. The return
  runner rejects a stale, replaced, or partially rebuilt artifact even if an old JSON file still
  says `complete`.
- **Market-input lineage clarification before any return was loaded:** the runner writes one
  canonical input manifest that binds the ticker-history snapshot; each actually loaded symbol's
  exact timestamp/open/close/volume values and missingness; the SPY-derived session calendar; and
  the normalized split/dividend rows consumed by the adjustment engine. The result binds that
  manifest's SHA-256, so unrelated lake partitions are excluded without weakening reproducibility.
- **Pair-integrity clarification before any return was loaded:** the pair builder recomputes every
  section's SHA-256 from its stored text, reconciles the stored word count, requires the locked
  parser version, and rejects malformed or non-increasing acceptance timestamps. It publishes an
  exhaustive adjacency attrition ledger: every latest-attempt corpus row is either an accepted
  immediate-predecessor pair endpoint or assigned a deterministic rejection reason. An unusable
  immediate predecessor is never skipped in favor of an older convenient filing.

The corpus may begin in 2005 to form predecessor pairs. Returns before 2016 are calibration only.
The locked OOS interval is 2016-01-01 through 2025-12-31.

## Signal

1. Extract Item 1A with `sec-filing-sections-v2`. Pair each filing only with the immediately prior
   unamended 10-K for the same CIK. Require both sections to contain at least 500 words.
2. Raw stability is five-token-shingle Jaccard similarity. Higher means less narrative change.
   No stemming, synonym model, length adjustment, or document-frequency weighting is allowed.
3. Form monthly cohorts by SEC acceptance calendar month. At the entry date, require unadjusted
   close at least $5 and median trailing 21-session dollar volume at least $5 million.
4. For each filing, compute the filing-reaction control from the adjusted close immediately before
   acceptance through the first complete XNYS close after acceptance, less SPY over the same
   interval. **Timing clarification before any return was loaded:** the end is the close of the
   first session whose opening timestamp is later than SEC acceptance; the start is the latest
   session close before acceptance. This is the preregistered substitute for unavailable
   point-in-time consensus surprise.
5. Compute 12-1 momentum at the cohort month-end as adjusted close at session -21 divided by
   adjusted close at session -252 minus one. Values must be knowable by that close.
6. Within each monthly cohort, percentile-rank stability, filing reaction, and momentum. Regress
   ranked stability on an intercept, ranked reaction, ranked momentum, and one-hot accession-time
   two-digit SIC. The signal is the OLS residual. No coefficient is carried across cohorts.
7. Require at least 20 eligible issuers and at least five names in each tail. Long the top residual
   quintile and short the bottom residual quintile. Ties break by CIK. A deficient cohort is flat.
   The one-hot regression must retain at least 10 residual degrees of freedom and produce at least
   10 distinct finite residuals; otherwise the cohort is also flat rather than ranked on numerical
   noise from a saturated industry design.

No SF1 outcome is included in the signal. The already-observed filing reaction is the locked
earnings-information control, and entry is delayed until that reaction is fully known.

## Timing and portfolio

- Enter each cohort at the **second XNYS session open after calendar month-end**. This ensures the
  first full post-acceptance close is known even for a filing accepted on the final day of a month.
- Hold each cohort for exactly 63 sessions. Cohorts do not refresh, restart, or exit on later news.
- Each cohort is equal notional, 50% long and 50% short. Average concurrent cohort weights and
  normalize stock gross exposure to 1.0. Positions shared across cohorts net before costs.
- Hedge trailing 252-session SPY beta, estimated only from adjusted close returns ending at the
  cohort month-end. Estimate each selected stock's beta, clamp each to [-1.0, 1.0], and equal-weight
  those betas using the cohort's signed stock weights. Aggregate active cohort stock weights and
  use their resulting beta for the SPY hedge. Normalize total gross including hedge to 1.0.
- Baseline costs are 15 bps one-way for stocks, 1 bp for SPY, and 3% annualized borrow on every
  short. Stress uses 30 bps, 2 bps, and 6% respectively. Costs apply to actual netted turnover.
- **Missing-open execution clarification before any return was loaded:** a stock target may change
  only on a session with an observed opening print for that stock. If an internal halt spans an
  intended rebalance or exit, the prior stock weight and its beta contribution remain held until
  the next observed open; the reopening interval is therefore realized before turnover cost. The
  SPY hedge follows the actually executable stock beta, and deferred target changes are reported.
- Report capacity at 1, 5, and 10 bps of trailing 21-session median dollar ADV; no stock position
  above 1% ADV is admissible.

CIK is mapped to the ticker row whose Sharadar first/last-price interval contains the entry date.
If no row or multiple issuer histories claim the same ticker on that date, the filing is excluded;
the runner may not guess from the current ticker. SPY uses the repository's pinned Yahoo
total-return research series. A cohort whose exact 63-session exit falls after 2025-12-31 is
excluded rather than truncated.

## Evaluation

The 2006–2015 calibration interval may expose broken mappings, impossible timestamps, missing
delistings, or unstable extraction, but cannot change a parameter or promote the sleeve. The
2016–2025 OOS curve is opened once after lineage checks, unit tests, and deterministic reruns pass.
Persist the full curve even if the candidate fails.

Report net and stressed Sharpe, Newey-West mean t-stat, DSR against the union experiment count,
drawdown, skew, turnover, beta, long/short contribution, annual results, capacity, ordinary and
bottom-decile-stress correlation to each current ALPHAC sleeve, and fixed-weight combined-book
delta. PBO is not estimated from a one-configuration surface; it is reported as not defined, never
as zero. The fixed marginal test assigns the candidate 10%, funded pro rata from the four current
equal-quarter sleeves (22.5% each). Also report a mean-zero candidate control and every
leave-one-calendar-year-out recomputation.

**Diversification-engine clarification before any return was loaded:** all correlation and
fixed-weight book evidence is computed by the shared
`alphaforge.validation.diversification.diversification_report` implementation. The common window
starts at the latest first observation and ends at the earliest last observation across the
candidate and four current sleeves; any missing date inside that window fails closed rather than
being dropped. The predeclared stress mask remains the bottom decile of the existing equal-weight
ALPHAC book. One-sided 95% correlation bounds use a deterministic 2,000-sample, 21-observation
circular moving-block bootstrap with seed 20260816. The report is persisted and SHA-256 bound.

## Kill and escalation rules

Kill if any condition holds:

1. OOS net Sharpe < 0.40, DSR < 0.95, Newey-West t-stat < 2.0, or stressed-cost Sharpe < 0.40.
2. Realized absolute SPY beta > 0.10.
3. Average ALPHAC correlation > 0.15, any ordinary pair correlation > 0.35, or any stressed
   correlation > 0.50. Also kill if the one-sided 95% upper bound exceeds 0.35 ordinarily or 0.50
   under stress, if fewer than 252 aligned observations or 63 stressed observations remain, or if
   internal common-window dates are missing.
4. The 10% candidate allocation fails to improve ALPHAC Sharpe, its mean-zero control, or any
   leave-one-year-out recomputation.
5. Either long or short stock leg has non-positive gross contribution.
6. Capacity at the 1% ADV ceiling is below $5 million.
7. The fixed 10% allocation worsens book expected shortfall, worsens maximum drawdown by more than
   one percentage point, or fails the shared canonical diversification artifact/hash contract.

If all research gates pass but historical locate/borrow evidence cannot be obtained, classify the
result `DATA-ESCALATE`, not `ADD`. Alpaca credentials are relevant only after an add-to-shadow
decision; they are not research data and cannot repair a failed backtest.

```prereg
profile: earnings_narrative_change_v1
lake_dir: data/lake_sharadar
alpha_names: sec_10k_item1a_stability_jaccard5
allocator: monthly_residual_quintile_beta_hedged
section: 10-K Item 1A
parser_version: sec-filing-sections-v2
direction: long_stable_short_changed
hold_sessions: 63
oos_start: 2016-01-01
oos_end: 2025-12-31
```

## Primary references

- SEC EDGAR APIs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- SEC access and archive paths: https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data
- Cohen, Malloy, and Nguyen, *Lazy Prices*: https://www.nber.org/papers/w25084
