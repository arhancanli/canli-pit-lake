# Pre-Registration — PIT Macro Surprise Family (5 remaining series)

**Written 2026-08-05, BEFORE any of these five was run.**
The git commit hash of this file is its timestamp of record.

---

## Why this family is being opened

`infl_surprise_size` (CPI headline + core → IWM/SPY size rotation) cleared all four
pre-registered gates: net Sharpe +0.340, NW t +1.82, placebo on QQQ−SPY dead at t −0.10,
ρ to the live book −0.069. One validated mechanism.

We hold point-in-time vintages for **seven** macro series, not two. Five are untested:

| Series | What it is | Vintages | Span |
|---|---|---|---|
| `EMPLOY` | nonfarm payroll employment | 739 | 1964–2026 |
| `IPT` | industrial production | 764 | 1962–2026 |
| `HSTARTS` | housing starts | 701 | 1968–2026 |
| `RUC` | unemployment rate | 243 | 1965–2026 |
| `RCONM` | real personal consumption | 333 | 1998–2026 |

The tradable window is bounded by ETF price history (SPY from 1993, IWM from 2000), not by
the vintages, so all five test on roughly the same 2001+ window as CPI did.

## The honest cost

Five runs is **five trials**. They enter `var/experiments.jsonl`, raise N, and push the
expected-max-Sharpe hurdle SR\* up for every future candidate including deployed ones. Hence
the declaration, in full, before execution:

* **All five are declared now.** No result may be reported without the other four.
* **No series may join this family after a result is seen.** A later idea is a new family.
* **No re-running, no re-windowing, no alternative spreads, no parameter variation.**
* **Bonferroni**: with 5 trials, a nominal one-sided p of 0.05 requires p ≤ 0.01, i.e.
  **NW t ≥ 2.33** for family-wise significance. A series clearing only t ≥ 1.5 is *suggestive*
  and earns a pre-registered forward test, never funding.

## The specification — identical to the validated CPI config

Every parameter is inherited unchanged from `scripts/probe_cpi_surprise_size.py`. Nothing is
re-tuned per series; that is the entire point of a family test.

| Field | Value |
|---|---|
| Instruments | IWM and SPY only |
| Surprise | expanding-window AR(3) residual of the newly-added observation, fit strictly on history **before** that observation, standardized by in-sample residual sd, clipped ±3 |
| Growth rates | log-differences computed **WITHIN a single vintage column** — never differencing `first_release` across vintages (mixes base-year rebasings; the lake's own DATA_CONTRACT documents this trap) |
| Weight | `w = SIGN × clip(SI, −1, +1)` on the dollar-neutral (IWM − SPY) spread |
| Timing | enter at the close of the first trading day **strictly after** the vintage date; hold to the next vintage date |
| Costs | 6bp one-way per leg |

### The signs are declared NOW, from mechanism, not fitted

This is the part that would be trivial to cheat, so it is fixed in advance:

* **Inflation channel** — a higher-than-expected price level is *bad* for small caps
  (margin pressure, rate sensitivity, weaker balance sheets). CPI used `SIGN = −1`, validated.
* **Growth channel** — a higher-than-expected level of *activity* is *good* for small caps
  (they are more cyclical and more domestically exposed). So `EMPLOY`, `IPT`, `HSTARTS`,
  `RCONM` all take **`SIGN = +1`**.
* **`RUC` is inverted by construction** — a higher-than-expected *unemployment rate* is worse
  activity, so it takes **`SIGN = −1`**.

Note this is a genuinely opposite prediction to the CPI result, and it is grounded: measuring
the CPI candidate, rising 2y yields went *with* small-cap outperformance (corr +0.155) — the
growth channel — while the inflation channel ran the other way. If a growth series comes back
significantly negative, that is a **failed prediction**, not a discovery to be re-signed.

## The redundancy gate — likely the binding constraint

Macro surprises plausibly measure the same underlying economy. Employment, industrial
production and consumption may be near-substitutes, in which case five passing series is
**one sleeve, not five**.

So, mandatory and pre-declared:

1. Compute the pairwise correlation matrix of the five resulting **strategy return series**,
   and each against `infl_surprise_size` and against the four live sleeves.
2. Any survivor with **|ρ| > 0.30 to an existing sleeve or to another survivor** is not counted
   as a separate sleeve. Within a correlated cluster, only the single highest-t member is kept.
3. This is the rule that killed `eq_noa` (ρ 0.32–0.44 to Investment) and
   `eq_cash_equity_issuance` (ρ 0.845 to a factor already run). It applies identically here.

A near-zero mean disqualifies regardless of correlation: adding a sleeve with own Sharpe +0.067
and ρ≈0 raised effective breadth to 5.61 — the best we have ever measured — and *lowered* book
Sharpe from 0.864 to 0.831.

## Reading rules

Per series, the same four gates the CPI config passed: (a) own_SR > ρ × S_b, (b) NW t ≥ 1.5,
(c) the benefit is the mean not variance dilution, (d) the QQQ−SPY placebo stays insignificant.

Then the redundancy gate above, applied to whatever survives.

## Prior, stated so it cannot be adjusted afterwards

Nine sweeps have produced two candidates. I expect **most of these five to be nulls**, and I
expect any survivors to be **substantially correlated with each other**, because they measure
one economy through different instruments. The realistic good outcome is *one* additional
uncorrelated sleeve from this family — not five. Zero is entirely plausible and publishable.

A null here is also informative in its own right: it would say the CPI result is specifically
an *inflation* effect rather than a generic "macro surprises move the size spread" effect,
which materially raises confidence that the CPI finding is real rather than a fluke of the
size spread being predictable by anything.


## Machine-checkable declaration

The block below is the ENFORCED contract. `alphaforge.validation.prereg.assert_matches`
reads it and kills any run whose resolved settings disagree, before compute is spent.
Added 2026-08-07 after three runs used the wrong lake: two burned a trial and returned a
silent null, one crashed after four hours. Every declaration was correct; nothing read it.

```prereg
profile: equity
lake_dir: data/lake_macro_vintage
allocator: none
```
