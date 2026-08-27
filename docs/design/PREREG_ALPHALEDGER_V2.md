# PRE-REGISTRATION — AlphaLedger v2 (`eq_asset_growth` with an explicit liquidity floor)

> ## ⛔ UNEXECUTABLE — declared void 2026-08-08, before any result was seen
>
> This configuration was run once and **crashed identically to v1**, on the correct lake, with the
> liquidity floor applied. The floor cannot work: it filters a STATIC full-history median dollar
> volume while the engine caps orders against a ROLLING 30-session median. It measures the wrong
> statistic, so it cannot prevent a transient condition.
>
> The underlying blocker is an engine limitation, not a universe or data problem — see
> `FINDING_WALKFORWARD_ADV_HALT.md`. Over 26 years and ~6,500 names, some held name having a thin
> 30-day window is a certainty, and the engine halts the entire run on the first one.
>
> **This document is NOT being amended.** Loosening or re-specifying the floor after four failed
> runs, to make a run pass, is precisely what its own "what would make this dishonest" section
> prohibits. It stands as written, marked unexecutable, and no result was produced from it.
>
> **No trial was consumed** — the run never completed, and the ledger is unchanged at 134 rows.


**Declared 2026-08-08, BEFORE this configuration was run.** Written as a NEW document rather than
an amendment, because the thing it changes is the one thing its predecessor forbade changing.

## Why v1 cannot be executed, and why that is not a technicality

`PREREG_SLEEVE4_INVESTMENT.md` pins the universe to the frozen
`universe_allowlist_20260619.json` cohort and states: *"No parameter below may be changed. If any
of it is altered, this is a new trial and this document is void."*

Executed faithfully on 2026-08-07, that configuration **crashes**, three times, identically:

```
CostModelMisuse: sqrt impact law invalid: notional 132.0 exceeds 5% of ADV 434.0
                 (pre-trade checks should have rejected this at 1% of ADV)
```

The cause is not the engine and not the lake — the same failure reproduced under both the `equity`
and `sharadar` profiles. **Roughly 3% of the pinned cohort is untradeable**: names whose 5th
percentile daily dollar volume is so low that 1% of ADV cannot buy a single share. Measured
examples: `LOTZ` and `MDVN` have stretches at **$0/day**; `ORIG` trades **$21/day** against a
$15 share price. When the engine meets one, the 1% participation cap floors to zero lots and it
deliberately returns the order unclamped so the cost model halts — `# sub-lot cap: let the 5%-ADV
tripwire halt loudly` (backtest/engine.py:1029). That is the engine refusing to pretend it can
trade a stock with no volume, which is correct behaviour.

So v1 pins a universe and a cost model that are **mutually incompatible on part of the cohort**.
It is not reproducible, and its published evidence (21y Sharpe 0.83, NW t +3.19, ρ −0.367 to
AlphaMax) rests on a run whose universe differed from its own declaration in both directions.

v1 is therefore **VOID BY ITS OWN TERMS**, and this document does not attempt to rescue it.

## The single change

**A point-in-time liquidity floor.** At each rebalance, an instrument is eligible only if its
**trailing 60-session median dollar volume is at least $250,000**, computed from bars strictly
before the decision bar.

* **PIT by construction.** Trailing-window only. A name that becomes liquid later is not eligible
  earlier, and a name that dies is dropped when it dies rather than in hindsight.
* **$250,000 because 1% of it is $2,500**, comfortably more than one share of anything in the
  cohort, so the engine's participation cap can always clamp instead of halting.
* **It removes ~4.3% of the cohort** (measured on a 600-name sample: $250k keeps 95.7%). It is a
  tradability filter, not a performance filter, and it is applied blind to returns.

**Everything else is inherited from v1 unchanged**: signal `eq_asset_growth` = `assets_t /
assets_{t-4q} − 1` at direction −1, allocator `rank`, train/test/purge/embargo 252/63/63/274,
rebalance 63 sessions, no-trade band 0.001, house cost model (6bp one-way, 50bp/yr borrow), the
same frozen cohort as the starting set, and the Sharadar SF1 lake as the source.

## The honest cost

**This is ONE new trial.** It enters `var/experiments.jsonl`, raises honest N from 133 to 134, and
lifts the deflation hurdle for every sleeve already in the book. That is the correct price for
turning an unrunnable declaration into a runnable one, and it is stated here rather than hidden
behind the word "re-run".

## Kill conditions, declared in advance

**K1 — it must actually run.** A crash is a FAILURE of this pre-registration, not a reason to
loosen the floor. If $250k is insufficient, that is a finding to publish, not a knob to turn.

**K2 — net Sharpe ≥ 0.30** on the full 2000–2026 walk-forward. Below that it does not clear the
bar the kill log has applied to every other candidate.

**K3 — book contribution.** Added at equal weight to the four-sleeve book (AlphaForge, AlphaMax,
AlphaTrend, AlphaVintage), it must **raise** book Sharpe. Measured 2026-08-07, that book is
**1.4223**. `eq_net_issuance` scored +0.2304 standalone and took the book DOWN to 1.2542; a
positive standalone Sharpe is not sufficient and this project has now demonstrated that twice.

**K4 — the v1 comparison must be published whichever way it falls.** v1's artifact reported ρ
−0.367 to AlphaMax and a +0.145 book contribution. If v2 measures materially worse, the honest
reading is that v1's number was an artifact of an untradeable universe, and it gets retracted in
public.

## What would make this dishonest

Loosening the floor after seeing a result; reporting v2 without v1 beside it; describing this as a
"re-run" when it is a new trial; or keeping v1's headline numbers anywhere after v2 measures.

## Prior

Stated now so it cannot be adjusted afterwards. `eq_asset_growth` screened weakly (t_nw +0.829,
IC +0.0117, rank far below `eq_net_issuance`'s +3.542) and net-issuance then destroyed book value.
Removing ~4.3% of the cohort should matter little if the effect is real and a lot if it was
carried by illiquid microcaps. **Honest expectation: 35% that it clears K2 and K3 together.** A
clean null is a perfectly good outcome and will be published as one.

## Machine-checkable declaration

The block below is the ENFORCED contract, read by
`alphaforge.validation.prereg.assert_matches` before any compute is spent.

```prereg
profile: sharadar
lake_dir: data/lake_sharadar
alpha_names: eq_asset_growth
allocator: rank
universe_allowlist: data/research/universe_allowlist_20260619.json
universe_sha256: 2fd82d305a777a92591e5e97ff47c036a665f70e86baf4bb5cfec1c16bb76cee
min_median_dollar_adv: 250000
adv_lookback_sessions: 60
```
