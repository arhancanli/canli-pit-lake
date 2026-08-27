# Pre-Registration — Fundamental Factors, Measured Individually

**Written 2026-08-04, BEFORE any of these eight runs was executed.**
The git commit hash of this file is its timestamp of record.

---

## Why this family is being opened at all

The original pre-registration tested fundamentals as **composites**, and both failed:

* slot 2 — "Value (composite)" → net Sharpe −0.5994, killed
* slot 3 — "Quality (GP/A + ROE)" → net Sharpe −0.8316, killed

A failing composite does not mean its components fail. Averaging a working factor with two dead
ones produces a dead average, and the working one is never seen. We now have direct proof of
exactly that: `eq_asset_growth` is the only fundamental factor this fund ever measured **alone**,
and it is the one that worked (21y Sharpe 0.83, NW t +3.19, ρ to book −0.369).

`src/alphaforge/features/library/equity_fundamental.py` implements eleven factors. Two were
measured as composites and killed. One was measured alone and adopted. **Eight have never been
measured at all**, including two with strong independent literature — accruals (Sloan 1996) and
net issuance (Daniel–Titman; Pontiff–Woodgate) — both of which have accounting-identity or
forced-flow mechanisms rather than being pure risk premia.

## The honest cost of running this

Eight runs is **eight trials**. They enter `var/experiments.jsonl`, they raise N, and they push
the expected-max-Sharpe hurdle SR\* up for every future candidate including the ones already
deployed. This is not free and it is not being pretended to be free.

That is precisely why the family is declared here, in full, before execution:

* **All eight are declared now.** No result may be reported without the other seven.
* **No factor may be added to this family after seeing a result.** If a ninth idea occurs later,
  it is a new family with its own document.
* **No re-running, no re-windowing, no sign-flipping, no parameter variation.** One config each,
  fixed below. A factor that fails is killed and published in the kill log.
* **Bonferroni is applied.** With 8 trials in the family, a nominal one-sided p of 0.05 requires
  p ≤ 0.00625, i.e. **NW t ≥ 2.50** to claim family-wise significance. A factor clearing only
  t ≥ 1.5 is reported as *suggestive, not established* and does not get funded on this evidence
  alone — it would require a pre-registered forward test like sleeve #4's.

## The eight factors

Every one is run with the **identical** configuration used for `prereg_investment`, so the family
is a clean like-for-like comparison and no factor gets a bespoke setup.

| # | Factor | Mechanism | Literature |
|---|---|---|---|
| 1 | `eq_accruals` | accounting identity: accrual-heavy earnings are lower quality and mean-revert | Sloan (1996) |
| 2 | `eq_net_issuance` | forced flow: firms issue when overvalued, buy back when undervalued | Daniel–Titman; Pontiff–Woodgate |
| 3 | `eq_gross_profitability` | profitability premium, gross profits to assets | Novy-Marx (2013) |
| 4 | `eq_book_to_price` | classic value | Fama–French |
| 5 | `eq_earnings_yield` | value, earnings-based | Basu |
| 6 | `eq_sales_to_price` | value, sales-based (robust to accounting noise) | Barbee et al. |
| 7 | `eq_operating_margin` | profitability/quality | — |
| 8 | `eq_roe` | profitability/quality | Haugen–Baker |

## Frozen configuration (identical for all eight)

| Field | Value |
|---|---|
| Universe | frozen `universe_allowlist_20260619.json` cohort (sha256 `2fd82d30…`) |
| Allocator | `rank` |
| Train / test | 252 / 63 days |
| Rebalance | 63 bars (quarterly) |
| Purge / embargo | 63 / 274 bars |
| No-trade band | 0.001 |
| Cash | $100,000 |
| Window | 2000-01-01 → 2026-06-01 |
| Costs | house model: 6bp one-way, 50bp/yr borrow on short gross |
| Fundamentals | Sharadar SF1 lake (frozen 2026-06-20) — the same source `prereg_investment` used |

## Pre-declared reading rules

1. **Report all eight**, ranked, with NW t and correlation to the existing book. Nulls published.
2. **Redundancy check before enthusiasm.** Any survivor must be checked for correlation to
   `eq_asset_growth`. Several of these are mechanically related to asset growth (net issuance and
   external financing are both investment-funding signals). A factor that merely proxies sleeve #4
   adds nothing to the book however good its standalone Sharpe looks, and must be rejected.
3. **A near-zero mean disqualifies regardless of correlation.** Measured today: adding a sleeve
   with own Sharpe +0.067 and ρ≈0 raised effective breadth to 5.61 — the best available — and
   *lowered* book Sharpe from 0.864 to 0.831. Decorrelation without edge destroys value.
4. **Nothing here is funded off this run.** The most a survivor earns is the right to a
   pre-registered forward test.

## Prior

Stated now so it cannot be adjusted afterwards: seven sweeps have returned seven nulls, and the
composites containing several of these factors already failed. The honest expectation is that
**most or all of these eight will be nulls**. One survivor would be a good outcome. Zero survivors
is a perfectly plausible and publishable result.


## Machine-checkable declaration

The block below is the ENFORCED contract. `alphaforge.validation.prereg.assert_matches`
reads it and kills any run whose resolved settings disagree, before compute is spent.
Added 2026-08-07 after three runs used the wrong lake: two burned a trial and returned a
silent null, one crashed after four hours. Every declaration was correct; nothing read it.

```prereg
profile: sharadar
lake_dir: data/lake_sharadar
allocator: rank
universe_allowlist: data/research/universe_allowlist_20260619.json
universe_sha256: 2fd82d305a777a92591e5e97ff47c036a665f70e86baf4bb5cfec1c16bb76cee
```
