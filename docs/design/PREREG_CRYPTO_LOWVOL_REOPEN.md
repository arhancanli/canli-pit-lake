# PRE-REGISTRATION — reopening `crypto_lowvol_720` on the correlation axis

**Declared 2026-08-06, BEFORE any correlation was computed.** Written first precisely because the
result is not yet known and the conclusions below must not be reachable by choosing the analysis
after seeing the answer.

## Why this is being reopened

`crypto_lowvol_720` was killed on **deflation: DSR 0.04** (`scripts/glassbox_export.py:221`). Net
walk-forward Sharpe **0.6947** over 1,248 days (2023-01-01 → 2026-06-01), vol 13.0%, max DD 20.1%,
turnover ~20x/yr. It was the strongest in-sample signal the 200-factor campaign ever produced
(Rank-IC t = 6.66).

Two facts, both established 2026-08-06, make that kill worth re-examining. Neither is a claim that
the sleeve is good.

1. **The DSR bar it failed is one every incumbent also fails.** Re-derived at honest N=127 and
   pooled V[SR] = 7.96e-04: AlphaMax **0.213**, AlphaForge **0.052**, AlphaTrend **0.000**. Against
   a 0.95 gate, none of the three sleeves in the live book clears it either. A candidate rejected
   at 0.04 while an incumbent at 0.052 is retained is not a standard being applied, it is an
   incumbency advantage. That was disclosed publicly today.
2. **Its correlation to the book has never been computable.** `artifacts/walkforward/crypto_lowvol_720/`
   contains exactly one file, `summary.txt`. The equity curve was never persisted. Since a sleeve's
   value in a portfolio is its *uncorrelated* return and not its standalone significance, the single
   number that decides this candidate has never been measured.

**This re-run costs ZERO new trials.** The config (`alpha_names: ['lowvol_720']`, `allocator: rank`,
`rebalance_bars: 63`) already exists in `var/experiments.jsonl` and is already counted in N=127.
Re-executing it to persist an artifact is not a new hypothesis. If any step below requires a config
change, that step becomes a NEW trial and must be declared separately before it is run.

## The measurement

Re-run the existing config unchanged, persist `equity.parquet`, and compute daily log returns.

**The lag-alignment requirement (mandatory).** Correlating a rebuilt series against a published
sleeve curve on a naive calendar join was measured today to understate correlation by ~30x
(−0.029 vs +0.871 on the same two series), because published curves are stamped one session after
the day they represent. Every correlation below MUST be computed at the lag that maximises |ρ|
across {−1, 0, +1}, and the chosen lag must be reported. Reporting only the calendar-join figure is
the single most likely way this analysis produces a false pass.

## Kill conditions — declared in advance, evaluated in this order

**K1 — book correlation.** ρ to the equal-risk combination of the three real sleeves
(AlphaMax `k30_dn_63`, AlphaTrend `mf_live_fwd`, AlphaForge `crypto_carry_wk`), on ≥500 overlapping
days, at the lag-aligned maximum. **ρ > +0.15 → KILL.**

**K2 — the costume test.** ρ to `crypto_carry_wk` specifically. **ρ > +0.25 → KILL**, on the ground
that it is the existing crypto sleeve wearing a different name. One prior worth stating in advance,
which is a reason for interest and NOT evidence: the summary reports `funding_net −11,303`, i.e.
this strategy *pays* funding while AlphaForge *earns* it, so they sit on opposite sides of the same
trade. That is a hypothesis about why ρ might be low. It is not a measurement and must not be
treated as one.

**K3 — the single-factor test.** Regress daily returns on BTC returns and on a BTC-dominance proxy.
**R² > 0.50 → KILL**: it is one macro bet across one regime rather than a cross-sectional premium,
which is the objection raised against it and which has never been checked.

**K4 — contribution.** Using measured ρ and Sharpe, compute the book-Sharpe contribution at
equal weight. **Contribution < +0.024 → KILL** (that is the deflation charge of an ordinary
15-trial search; a candidate that does not beat it is not worth a slot even when free).

**K5 — venue reality.** The backtest was built on Binance data and Binance is unreachable from this
network. If K1–K4 all pass, net Sharpe must be re-derived under OKX costs and the OKX-available
universe before any deployment. A pass on Binance data is NOT a pass to trade.

## If it survives

It enters at **10% of full weight** on the probation ladder and earns more only from live forward
performance. It is never described as "cleared" or "deflation-proof" — its DSR is 0.04 and that does
not change. The honest description is: a candidate admitted on measured decorrelation, at token
weight, whose forward record is the only thing that can promote it.

## If it dies

The kill is published with the number that killed it, and the sleeve closes permanently. A candidate
that fails a pre-registered test does not get re-tested on a different axis.

## What would make this pre-registration dishonest

Changing any threshold above after seeing a result; reporting a calendar-join correlation instead of
the lag-aligned one; dropping K3 because K1 and K2 passed; or re-running with a modified config and
describing it as the same zero-trial re-execution.


## Machine-checkable declaration

The block below is the ENFORCED contract. `alphaforge.validation.prereg.assert_matches`
reads it and kills any run whose resolved settings disagree, before compute is spent.
Added 2026-08-07 after three runs used the wrong lake: two burned a trial and returned a
silent null, one crashed after four hours. Every declaration was correct; nothing read it.

```prereg
profile: base
lake_dir: data/lake
alpha_names: lowvol_720
allocator: rank
```
