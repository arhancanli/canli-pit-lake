# Pre-registration: crypto_carry_portable_v1

**Author and accountable investigator:** Arhan Canli  
**Frozen:** 2026-08-24, before portable-v1 return computation  
**Research epoch:** prospective v7  
**Identity:** new prospective return identity; not a replication of the historical crypto-carry result

```prereg
profile: base
lake_dir: var/portable_crypto_carry_v1/lake
alpha_names: carry_fund_21
allocator: rank
return_identity_id: crypto_carry_portable_v1
instrument_count: 57
train_bars: 6048
test_bars: 1512
rebalance_bars: 168
```

## Question and falsifiable hypothesis

The question is whether a weekly cross-sectional funding-rate carry signal, implemented through the
current AlphaForge code and current checksum-verified official archive rows, produces useful
out-of-sample net returns after the declared transaction-cost, execution, risk and anti-overfitting
controls.

The economic mechanism is compensation paid by persistently crowded perpetual-futures positioning.
At each decision, the strategy ranks the lagged funding-carry signal across the point-in-time liquid
universe and takes a constrained long/short allocation. The falsifiable alternative is that the
observed funding spread is too unstable, too correlated with crash risk, or too costly to monetize.
Failure is an accepted scientific outcome and receives the same permanent packet as success.

No portable-v1 return, equity curve, Sharpe ratio, drawdown, admission result or stress result had
been computed when this document was frozen. Historical crypto-carry returns were known, but they
were produced by a materially different, incompletely bound state and cannot be used to label this
new identity a replication.

## Identity and trial accounting

- Family trial account: `crypto_carry`.
- Return identity: `crypto_carry_portable_v1`.
- Exact hypothesis identity: `da5f5f47f99f9bd2`.
- Exact config hash: `50d9e8b059fee773`.
- Hypotheses charged if and only if the governed runner records the return result: one.
- The trial is ordinal 229 if no earlier forward identity is recorded first.
- The result, regardless of sign, must be completed and published as a full identity packet before
  another prospective identity is allowed to run.
- Re-running identical bytes is idempotent and may not increase the union count.

## Pre-result data decision

The source universe starts from the 58 instruments in the frozen portability inventory. Eligibility
is determined without return information: exclude every symbol with at least one unavailable
required official monthly archive object. All 14 unavailable objects are ICPUSDT funding archives
from July 2021 through August 2022; therefore ICPUSDT is excluded and exactly 57 instruments remain.
No frozen local market row or unbound API response may fill a fresh-source gap.

The exclusion preserves the original 2021-06-01 through 2026-06-01 window and 25 walk-forward legs.
The rejected alternative—delaying the start to September 2022 while retaining 58 instruments—would
leave 18 legs. This choice was made from source availability and statistical span only, before any
portable-v1 returns.

Current official archives differ from the frozen local lake on 1,256 overlapping OHLCV fields.
Current checksum-verified archive rows are authoritative for this trial. Consequently the trial is a
new current-state result, not an exact historical replication.

## Bound data and lineage

- Data contract: `config/crypto_carry_portable_v1_prerun.json`, content hash
  `sha256:e153521aa08d1ee1a27a64ff008e3ea25ba72858017133cb781178dcfb8b48e3`.
- Public data-readiness receipt: `artifacts/audit/crypto_carry_portable_lake_readiness.json`, content
  hash `sha256:d6ead5a3dd1b0db3f5db840ef3a4b00519d63b4af0ed354741cd33d0389f7d62`.
- Private manifest content hash:
  `sha256:7450ec650d611a55a8bd52159c1c23de21951764d68c3ee059a682a0b202651e`.
- Private leaf-inventory root:
  `5432c5fb344b500309c923714d14a65040da39c49f027f0e7769cd0cfc6785ea`.
- Retained input: 2,137,040 hourly OHLCV rows and 272,846 funding rows, all lifecycle-valid.
- Universe: 108 point-in-time liquidity-membership intervals reconstructed from the isolated lake.
- Instrument metadata: the bound 57-row subset of the frozen local SCD2 packet. Each retained current
  row is made valid from the run start for portable execution; its original `valid_from_ms` remains
  audit lineage. Fresh historical exchangeInfo was not reacquired, so this remains a stated
  limitation and prevents an independent-replication claim.
- Funding knowability: `available_at = ts_funding + 5 minutes`; consumers join on `available_at`.
- Before the first trade, the walk-forward runner must atomically snapshot the signal frame,
  point-in-time universe rows, instrument rows, consumed lake partitions, resolved settings, source
  tree and lockfile. Snapshot failure blocks the run.
- Raw archives and derived market rows remain private because redistribution rights are not
  established. Public evidence may contain hashes, counts, methods and aggregate results only.

## Frozen trial configuration

| Field | Frozen value |
|---|---:|
| Start, inclusive | 2021-06-01 00:00 UTC |
| End, exclusive | 2026-06-01 00:00 UTC |
| Training window | 6,048 hourly bars |
| Test window / step | 1,512 hourly bars |
| Expected OOS legs | 25 |
| Purge | 72 bars |
| Embargo | 168 bars |
| Signal | `carry_fund_21` |
| Allocator | rank |
| Rebalance | 168 bars |
| No-trade band | 0.001 of equity |
| Initial cash | $100,000 simulation capital |
| Covariance window | 720 bars |
| Covariance minimum | 240 bars |
| ML gate | off |
| Regime gate | off |

The exact 57 instrument IDs are frozen in `config/crypto_carry_portable_v1_run.json`; their order is
part of the hashed trial configuration. All non-path settings come from the hash-bound
`configs/base.yaml`. `AF_*` environment overrides are forbidden. The project and dependency
authorities are the hash-bound `pyproject.toml` and `uv.lock`.

## Execution and risk assumptions

The trial uses the production AlphaForge event-driven engine and production PIT readers. Orders fill
under the existing next-open model. The base configuration declares 5 bp perpetual taker fees, 2.5
bp default half-spread, a 2 bp latency add-on and the square-root impact model with a 1% ADV order
cap. Portfolio limits include 1.0 gross, 0.5 absolute net, 0.15 per-position weight, 0.15 annualized
volatility target and 1.5 maximum volatility scale. The risk state halves gross at a 10% drawdown,
flattens at 15%, and uses a 336-hour cooldown. These controls are implementation facts, not claims
that realized or expected drawdown will satisfy the 11% book objective.

## Outcomes and estimands

The primary result is the single stitched out-of-sample hourly equity path across the 25 purged
walk-forward legs. The packet must report, at minimum:

1. net annualized Sharpe and its observation count;
2. maximum drawdown, CAGR, annualized volatility, Sortino and Calmar;
3. total fees, net funding, annualized turnover and risk-state counters;
4. PSR and DSR against the complete hypothesis union in force when recorded;
5. per-leg summaries and leave-one-period-out stability;
6. PBO or an explicit blocker if the required path matrix is not generated;
7. baseline, stressed-cost and stressed-execution outcomes;
8. capacity at no fewer than three capital points;
9. ordinary and stressed correlation with the existing book, including uncertainty bounds; and
10. expected and 95th-percentile book drawdown contribution.

No unregistered variant may replace the primary result. Any code, data, universe, timing, allocator,
signal, rebalance, cost or risk mutation is a new identity unless the governing accounting contract
explicitly classifies it as a non-return diagnostic.

## Decision rule

Admission is governed by the v7 contract in force at reservation time; this document cannot weaken
it. The primary candidate must clear every applicable family, execution, capacity, stress,
diversification and book-level gate. Key thresholds include net and stressed Sharpe at least 0.15,
PBO no more than 0.20, measured DSR, at least 756 OOS observations, capacity of at least $500,000,
ordinary pairwise correlation no more than 0.35, average correlation to the existing book no more
than 0.00, and strictly positive book Sharpe contribution with the required uncertainty and
leave-one-period-out checks. The portfolio expected-maximum-drawdown objective remains at or below
11% and the 95th percentile must be published beside it.

Possible dispositions are:

- `ADMIT`: every required gate passes and the complete packet is sealed;
- `KILL`: a scientific, robustness, execution, capacity, stress, diversification or book gate fails;
- `INCOMPLETE`: required evidence cannot be computed, in which case admission is prohibited; or
- `INVALID`: preregistration, data, snapshot, code, environment or reservation binding fails.

A positive Sharpe does not override a failed gate. A negative result does not permit a gate change,
an instrument deletion, a shortened period or a relabeled identity.

## Publication commitment

The permanent public packet is reserved at
`/glassbox/trial-packets/crypto_carry_portable_v1.json`, and the paper at
`/research/crypto-carry-portable-v1`. Arhan Canli is the named author and accountable investigator.
The packet will include this preregistration, exact input and code bindings, complete result,
uncertainty, failed gates, reproducibility instructions and the historical correction boundary.
Repository preparation does not equal external submission, DOI assignment, peer review or
independent replication; those statuses may be claimed only after they actually occur.
