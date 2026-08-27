# Pre-Registration — Sleeve #4: Investment (Asset Growth)

**Written 2026-08-04, before the sleeve has traded a single share.**
The git commit hash of this file is its timestamp of record.

---

## 1. What is NOT clean about this candidate

This section comes first deliberately. A pre-registration that opens with its own strongest
claim is marketing; one that opens with its weakest link is a control.

**The in-sample result was already known when this document was written.** This candidate was not
drawn from an unspent pre-registered slot. It surfaced from a *retrospective screen* of 46
walk-forward artifacts we had already computed, run on 2026-08-04 while looking for candidates
that clear an allocation bar rather than the DSR discovery gate. Its historical numbers
(below) were visible before this document existed.

Therefore:

* This is **not** a clean pre-registered discovery and must never be described as one.
* Slots 8–9 of `PRE_REGISTRATION.md` are contingency reserves spendable "only against a
  structural trigger logged before its result is seen." That condition is **not** met here, so
  **no contingency slot is being spent**. This sleeve is declared outside the original budget and
  is labelled as such in the public record.
* The historical Sharpe is a **selected** number. Best-of-46 selection inflates it, and no
  amount of framing removes that.

**What IS clean, and it is the only thing being claimed:** the forward record starting at go-live.
It is out-of-sample by construction, it cannot be re-run, and it is signed into the transparency
chain from day one.

## 2. Why it was not dismissed outright

Three mitigations, stated as arguments rather than proof:

1. **It belongs to a pre-registered family whose siblings failed.** The `prereg_*` factor set was
   momentum, value, quality, betting-against-beta and investment. Four failed
   (−0.09, −0.50, −0.70, −0.10). One did not. That is 1-of-5 within a principled, theory-driven
   family — not 1-of-46 dredged from a parameter sweep.
2. **The evidence is long and strong.** 5,384 sessions (2005-01-04 .. 2026-06-01), PSR 0.9998,
   Newey-West t = +3.19. It fails DSR (0.00058) *only* on the 75-trial deflation penalty — the
   penalty that, at our trial count, no strategy below ~1.24 Sharpe can ever clear.
3. **The diversification is structurally expected, not fitted.** ρ(AlphaMax, Investment) = −0.369.
   Momentum and investment/value factors are negatively correlated in the literature and by
   construction; we did not go looking for a negative number and find one.

## 3. The frozen specification

No parameter below may be changed. If any of it is altered, this is a new trial and this document
is void.

| Field | Value |
|---|---|
| Signal | `eq_asset_growth` = `assets_t / assets_{t-4q} − 1`, direction **−1** (long LOW asset growth) |
| Theory | Fama-French CMA; Cooper–Gulen–Schill (2008) asset-growth effect |
| Universe | the frozen `universe_allowlist_20260619.json` cohort (sha256 `2fd82d30…`) |
| Allocator | `rank` |
| Rebalance | 63 sessions (quarterly) |
| Train / test / purge / embargo | 252 / 63 / 63 / 274 bars |
| No-trade band | 0.001 |
| Costs | unchanged house model: 6bp one-way, 50bp/yr borrow on short gross |
| Book weight at go-live | **10%**, funded from cash, not taken from an existing sleeve |

**Historical (selected, in-sample, NOT a forward claim):** Sharpe 0.83 · CAGR 6.0% · vol 10.9% ·
maxDD 19.0% · turnover 3.9x/yr · fees $1,126 per $100k over 21 years.

**Known weakness, stated now rather than after it bites:** on the recent 728-session window shared
with the live book, this sleeve's Sharpe is only **+0.27 (t = 0.49)**. The strength is in the long
history. The factor was crushed through the 2020–21 growth bubble (its 19% max drawdown runs
2020-06-09 → 2021-11-15). A forward Sharpe near zero is entirely consistent with the evidence and
would not, by itself, be surprising.

## 4. Data provenance, and the split that matters

* **History (validation):** Sharadar SF1, on disk, frozen at 2026-06-20. Not re-run.
* **Forward (live):** SEC EDGAR XBRL `us-gaap:Assets` via `scripts/ingest_sec_fundamentals.py`.

EDGAR is used **forward only**. Its ticker map covers current filers, so a historical rebuild from
EDGAR would be survivorship-biased. We therefore do not rebuild history from it, and no historical
number in this document comes from it.

Point-in-time rule, enforced in the ingest: for each `(ticker, period_end)` we keep the row with
the **earliest `filed` date** — the value as first disclosed, on the date it was first disclosed.
Restatements are discarded. Measured first-disclosure lag on the ingested sample: median 38 days
(p05 24, p95 73), consistent with 10-Q timing.

## 5. Pre-declared forward gates

Committed now, before any forward return exists. Evaluated on the live paper record only.

* **Review at 12 months and 24 months.** No evaluation before 12 months; interim numbers are
  noise and will not be used to justify a change.
* **KILL if**, at the 24-month review, the forward Sharpe is **< 0.0**. The sleeve is removed and
  the failure is published in the kill log.
* **KEEP at current weight if** forward Sharpe is between 0.0 and 0.3.
* **Weight may increase only if** forward Sharpe ≥ 0.3 **and** ρ to the book remains < 0.0.
* **No re-tuning under any outcome.** If it disappoints, it is killed or held — never re-fitted,
  re-windowed, or sign-flipped. A rescued parameter is a new trial and voids this document.

## 6. What gets published either way

The forward Sharpe, the realised correlation to the book, and the gate outcome — whether it
confirms or embarrasses us. This document is committed to git before go-live and anchored into the
signed transparency chain, so its content cannot be revised after the result is known.

The public description must state, in the sleeve's own entry: *"Surfaced by a retrospective screen
with its in-sample result already known; the historical Sharpe is a selected number. Only the
forward record is out-of-sample."*


## Machine-checkable declaration

The block below is the ENFORCED contract. `alphaforge.validation.prereg.assert_matches`
reads it and kills any run whose resolved settings disagree, before compute is spent.
Added 2026-08-07 after three runs used the wrong lake: two burned a trial and returned a
silent null, one crashed after four hours. Every declaration was correct; nothing read it.

```prereg
profile: sharadar
lake_dir: data/lake_sharadar
alpha_names: eq_asset_growth
allocator: rank
universe_allowlist: data/research/universe_allowlist_20260619.json
universe_sha256: 2fd82d305a777a92591e5e97ff47c036a665f70e86baf4bb5cfec1c16bb76cee
```
