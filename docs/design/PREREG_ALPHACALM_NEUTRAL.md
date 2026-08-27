# PRE-REGISTRATION — AlphaCalm, dollar-neutral variant

**Declared 2026-08-08, BEFORE this configuration was run.**

## Why the existing AlphaCalm cannot be deployed

`crypto_lowvol_720` passed a pre-registered reopening (K1–K4) on 2026-08-07 and is described as a
market-neutral cross-sectional low-volatility sleeve. Two things measured since say it is neither.

**1. It has never been neutral.** Across all 14 legs its net exposure averages **+24.5%**, median
+23.3%, MINIMUM **+13.6%**, reaching **+46.2%** in the final leg (10 names, 30% of NAV in gold
perps PAXG+XAU, shorting five illiquid micro-caps). AlphaForge, by comparison, averages +4.4%.

**2. The factor is not what its name says.** Ranking Binance perps by trailing 30-day realized
volatility at 2025-06-30, 2026-01-30 and 2026-05-30, the lowest-vol names are BTC, BNB, TRX, XRP,
LTC, ETH, ADA, SOL — **the majors**. "Cross-sectional low-vol" in crypto is a **SIZE ranking**, and
inverse-vol sizing then over-weights those low-vol majors, which is the mechanical origin of the
net-long tilt. Gold-pegged tokens sit lowest of all, which is why PAXG/XAU dominated the last leg.

**Root cause:** `dollar_neutral: true` is set ONLY in `configs/equity.yaml`. No crypto profile
enforces it, so no crypto sleeve has ever been constrained to neutrality.

So the sleeve as it stands is a long-biased large-cap crypto book. The book already buys crypto
beta deliberately and discloses it as a +20% strategic overlay; buying more of it under a
market-neutral label would be a misdescription, not a diversifier.

## The single change

**Enforce dollar-neutrality** on this sleeve, exactly as the equity sleeves already do. Everything
else is inherited from the reopened configuration unchanged: `alpha_names: ['lowvol_720']`,
allocator `rank`, rebalance_bars 63, no_trade_band 0.001, train_bars 8760 / test_bars 2184, window
2022-01-01 → 2026-06-01, and the same 3,112 instrument ids recorded in `var/experiments.jsonl`.

## The honest cost

**ONE new trial.** N 133 → 134. A config change is a new hypothesis; calling it a re-run would be
false.

## What this test is actually for

This is a decomposition, not a rescue. The reopened sleeve's measured book contribution came from a
book that was 24.5% net long on average. **If the edge survives enforced neutrality it is a real
cross-sectional effect. If it vanishes, the "edge" was crypto beta** — commoditized, already owned
via the disclosed overlay, and worth nothing as a separate sleeve. Either answer is informative and
both will be published.

## Kill conditions, declared in advance

**K1 — neutrality must actually bind.** Mean |net| across legs must be **< 5%** of gross. If the
constraint does not hold, the run is void and nothing is reported from it.

**K2 — net Sharpe ≥ 0.30** on the walk-forward.

**K3 — it must RAISE the four-sleeve book** (AlphaForge/AlphaMax/AlphaTrend/AlphaVintage, base
**1.4223**). `eq_net_issuance` (+0.148) and `eq_accruals` (+0.230) were both positive standalone and
both LOWERED it. Positive is not sufficient.

**K4 — the beta decomposition is published either way.** Report the neutral Sharpe beside the
long-biased one (+0.523 on a 365-day annualisation). If neutrality destroys the edge, that is the
headline result and the sleeve closes permanently.

## What would make this dishonest

Relaxing the neutrality constraint after seeing the result; reporting the neutral number without the
long-biased one beside it; or continuing to describe the original as "market-neutral" anywhere.

## Prior

Stated now so it cannot be adjusted afterwards. The low-vol ranking IS a size ranking, and size
premia in crypto are largely a beta story. **Honest expectation: 25% that it clears K2 and K3
together.** A null is the more likely outcome and will be published as one.

```prereg
profile: base
lake_dir: data/lake
alpha_names: lowvol_720
allocator: rank
dollar_neutral: true
```
