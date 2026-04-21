# Volatility Modelling & Market Crash Prediction

Replication and extension of Kim et al. (2011) using S&P 500 daily returns
(Jan 2001 – Mar 2020), implemented entirely from scratch in Python. Models
are evaluated on their ability to predict the March 2020 COVID crash (-12% 
single-day return).

## What this covers

**Volatility models estimated (6 total)**
- Constant Volatility (CV) — Normal and Student-t innovations
- GARCH(1,1) — Normal and Student-t innovations  
- ARMA(1,1)-GARCH(1,1) — Normal and Student-t innovations
  - No Python package supports joint ARMA-GARCH fitting; implemented
    manually via custom MLE with `scipy.optimize.minimize` (L-BFGS-B)

**Crash probability analysis**
- Estimated P(return ≤ −12%) for each model using one-step-ahead forecasts
- CV (Normal): probability ≈ 2.62 × 10⁻²³ — effectively impossible
- CV (Student-t): average recurrence time ≈ 1 year — plausible
- GARCH (Student-t): shortest recurrence at ~0.10 years (~25 trading days)
- Key finding: normality assumption catastrophically underestimates crash risk

**Goodness-of-fit**
- Kolmogorov-Smirnov tests on standardised residuals for all 6 models
- All models rejected at all significance levels — consistent with Kim et al.
- ARMA-GARCH (Student-t) achieved the lowest KS statistic (0.028), 
  indicating the closest empirical fit

**Rolling VaR and ES forecasting**
- 2,500-day rolling window, one-step-ahead forecasts for CV and GARCH models
- 99% VaR and Expected Shortfall computed analytically from model distributions
- GARCH models produce lower, more adaptive risk estimates than CV due to
  volatility clustering — CV overestimates risk in calm periods

**Backtesting — Christoffersen (CLR) and Berkowitz (BLR) tests**
- CLR unconditional coverage, independence, and joint tests
- BLR tail and independence tests via probability integral transform
- Only GARCH (Student-t) passes all CLR tests
- Only GARCH (Normal) passes both BLR tests
- All CV models rejected under both test families

**Average Relative Difference (ARD)**
- Quantifies how much Student-t vs Normal assumptions change risk estimates
- Student-t models consistently produce higher VaR and ES
- Effect is substantially larger in CV models than GARCH

## Key results

| Model | Innovation | Crash prob (−12%) | Avg time to crash | CLR result |
|---|---|---|---|---|
| CV | Normal | 2.62 × 10⁻²³ | 1.53 × 10²⁰ yr | Rejected |
| CV | Student-t | 0.397% | 1.01 yr | Rejected |
| GARCH | Normal | 1.71% | 0.23 yr | Rejected |
| GARCH | Student-t | 3.98% | 0.10 yr | **Passed** |
| ARMA-GARCH | Normal | 1.46% | 0.27 yr | — |
| ARMA-GARCH | Student-t | 1.85% | 0.22 yr | — |

## Tech stack

Python · NumPy · SciPy (`minimize`, `norm`, `t`, `kstest`, `chi2`) ·
`arch` library (CV and GARCH fitting) · Matplotlib · Pandas · yFinance

## Data

S&P 500 daily prices, January 10, 2001 – March 15, 2020 (5,083 trading days).  
Source: Yahoo Finance (`^GSPC`). End date chosen deliberately to evaluate
model predictions immediately before the COVID crash on March 16, 2020.

## Reference

Kim, Y.S., Rachev, S.T., Bianchi, M.L., Mitov, I. and Fabozzi, F.J. (2011).
Time series analysis for financial market meltdowns.
*Journal of Banking & Finance*, 35, 1879–1891.
