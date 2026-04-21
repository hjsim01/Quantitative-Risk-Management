# Extreme Value Theory & Aggregate Risk

Analysis of extreme tail risk in the Nikkei 225 index (2015–2025) using 
classical EVT methods, implemented from scratch in Python.

## What this covers

**Exercise I — Block Maxima & GEV Distribution**
- Extracted ~122 block maxima from 2,443 daily log returns (block size n = 20)
- Fitted GEV distribution via MLE; bootstrapped standard errors (1,000 resamples)
- Estimated 10-year return level with 95% CI using quantile inversion
- Bias-variance trade-off analysis across block sizes n = 10, 20, 40
  - Key finding: n = 40 significantly improved tail fit (ξ dropped from 0.77 → 0.32)

**Exercise II — GPD & Threshold Exceedances**
- Selected threshold u = 0.026 via sample mean excess plot (linearity criterion)
- Fitted GPD to 63 exceedances via MLE: ξ̂ = 0.270, β̂ = 0.00954
- Threshold stability analysis across 90th–99th percentile range
- Estimated VaR and ES at 99% and 99.5% confidence levels; validated 
  against empirical quantiles (< 0.05% deviation)

**Exercise III — POT Model & Risk Estimation**
- Constructed full POT model: exceedance rate λ̂ = 0.0258
- Scaled VaR and ES to k = 1, 10, 20-day horizons (independence assumption)
- Simulation study: 10,000 one-year scenarios (250 trading days) from POT model;
  compared simulated annual maxima against theoretical GEV via KS test
- Backtesting: Kupiec POF test on 20% holdout — 7 violations vs 4.89 expected,
  LR statistic = 0.81, p-value = 0.37 (model adequate at 1% level)

**Exercise IV — Coherent Risk Measures & Copula Dependence**
- Computed VaR and ES analytically for Student-t (ν=4), Normal, and GPD marginals
- Simulated aggregate portfolio loss (100,000 scenarios) under three dependence structures:
  independence, Gaussian copula (ρ=0.5), Student-t copula (ρ=0.5, ν=4)
- Verified ES subadditivity numerically; demonstrated VaR non-subadditivity 
  via bond default counterexample (McNeil, Frey & Embrechts 2005)
- Comonotonic VaR additivity confirmed: theoretical vs simulated discrepancy < 1.5%

## Key results

| Model | Method | Shape (ξ) | VaR₀.₉₉ (1-day) |
|---|---|---|---|
| GEV (n=40) | Block Maxima | 0.323 | — |
| GPD (u=0.026) | POT | 0.270 | 3.63% |
| Portfolio (independent) | Copula simulation | — | 3.23% |
| Portfolio (Student-t copula) | Copula simulation | — | 4.47% |

## Tech stack

Python · NumPy · SciPy (`genextreme`, `genpareto`) · Matplotlib · Pandas

## Data
Nikkei 225 daily prices, October 2015 – October 2025 (2,443 trading days).  
Source: Yahoo Finance 
