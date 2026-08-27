# PRE-REGISTRATION — AlphaLedger v3 (`eq_asset_growth`, floor on the ROLLING statistic)

**Declared 2026-08-08, BEFORE this configuration was run.** Third and final attempt; if it fails
its gates the sleeve is closed and published as a null.

## Why v1 and v2 failed, corrected

v1 pinned a universe with no liquidity condition. v2 added a floor on the **static full-history
median** dollar volume. Both crashed identically:

```
CostModelMisuse: sqrt impact law invalid: notional 132.0 exceeds 5% of ADV 434.0
```

**v2's floor measured the wrong statistic.** The engine caps each order at 1% of the decision
bar's **ROLLING 30-SESSION median** dollar volume, and halts when that floors below one lot. A name
with a $10M lifetime median can sit at a few hundred dollars for a month. Filtering the lifetime
median cannot prevent that.

I also claimed the engine simply cannot complete long equity runs. **That was wrong** —
`eq_net_issuance` and `eq_accruals` both completed all 86 legs on this exact window hours earlier.
The failure is specific to which names `eq_asset_growth` selects, not to the engine or the window.

## The declared filter

Exclude any instrument whose **rolling 30-session median dollar volume EVER falls below $25,000**
across the run window.

* **It matches the engine's own statistic** — same window length, same median, so the cap can
  always clamp to at least one lot instead of halting.
* **$25,000 because 1% of it is $250**, which buys a share of anything in this cohort.
* **It is conservative**: a name thin for one month in 26 years is excluded for the whole run, not
  just while thin. Stricter than a per-rebalance rule; it can only remove return, never add it.
* **It removes ~19% of the cohort** (measured on a 700-name sample: $25k keeps 81.3%). Stated here
  before running, because it is a much larger intervention than v2's 4.3% and that matters.

Everything else inherited unchanged: `eq_asset_growth` direction −1, allocator `rank`,
252/63/63/274, rebalance 63, band 0.001, house costs, Sharadar SF1 lake, the frozen cohort.

## Honest cost

**ONE new trial.** N 133 → 134. Stated plainly, not hidden behind "re-run".

## Kill conditions, declared in advance

* **K1 — it must run.** A further crash closes the sleeve. No fourth floor.
* **K2 — net Sharpe ≥ 0.30** on the full walk-forward.
* **K3 — it must RAISE the four-sleeve book** (AlphaForge/AlphaMax/AlphaTrend/AlphaVintage,
  base **1.4223**). `eq_net_issuance` (+0.148) and `eq_accruals` (+0.230) both LOWERED it. A
  positive standalone Sharpe is demonstrably not sufficient here.
* **K4 — v1's numbers are retracted regardless.** v1's 21y Sharpe 0.83 / NW t +3.19 / ρ −0.367
  came from a 2026-06-21 artifact on a pre-2026-08-04 engine and are not reproducible. Whatever v3
  measures replaces them; they are not quoted again.

## What would make this dishonest

A fourth floor after a fourth failure; reporting v3 without v1 beside it; or calling this a re-run.

## Prior

`eq_asset_growth` screened weakly (t_nw +0.829 vs `eq_net_issuance`'s +3.542), and both of its
better-screening siblings destroyed book value when measured. Removing 19% of the cohort is a real
intervention. **Honest expectation: 30% it clears K2 and K3 together.** A null is a fine outcome.

## Machine-checkable declaration

```prereg
profile: sharadar
lake_dir: data/lake_sharadar
alpha_names: eq_asset_growth
allocator: rank
universe_allowlist: data/research/universe_allowlist_20260619.json
universe_sha256: 2fd82d305a777a92591e5e97ff47c036a665f70e86baf4bb5cfec1c16bb76cee
min_rolling30_dollar_adv: 25000
```
