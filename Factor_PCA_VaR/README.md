# Factor Models, PCA Risk Attribution & VaR Backtesting

Portfolio risk analysis of a 5-stock tech portfolio (AAPL, MSFT, NVDA, AMD, INTC)
covering factor model estimation, PCA-based risk decomposition, VaR backtesting,
and multivariate distribution analysis. Implemented in Python.

## What this covers

**Exercise I — Two-Factor Market Model**
- Estimated X = α + β₁F₁ + β₂F₂ + ε for each stock against NASDAQ and S&P 500
- Key finding: NVDA (β₁ = 2.98) and AMD (β₁ = 2.41) are highly leveraged to
  NASDAQ; INTC is mostly idiosyncratic (R² = 0.29, 70% idiosyncratic risk)
- Only NVDA's α is statistically significant at 5% — evidence of abnormal returns
- Residual correlation matrix shows factor model eliminates most co-movement:
  cross-stock correlations drop from ~0.5–0.7 to near zero after factor removal

**Exercise II — PCA Risk Decomposition (Synthetic Portfolio)**
- Generated 12,000 observations from multivariate Student-t (ν=5, ρ=0.4,
  heteroskedastic variances) to simulate realistic heavy-tailed returns
- Computed VaR and ES at 90%, 95%, 99% confidence levels; ES/VaR ratio
  declines with α, showing diminishing marginal tail risk at extreme quantiles
- PCA on 6-asset returns: PC1 explains 53.1% of variance alone; 5 components
  needed for 90%, 6 for 95%
- Portfolio approximated using top 5 PCs (90% variance): VaR relative error = 0.22%
- VaR attribution: PC1 drives 51.2% of portfolio VaR — primary hedging target

**Exercise III — VaR Backtesting (3 methods × 2 confidence levels)**
- Equal-weighted portfolio of 5 tech stocks; 250-day rolling window
- Methods: Historical Simulation, Parametric Normal, Parametric Student-t
- Backtesting period: last 3 years (753 days), calibrated on prior data

| Alpha | Method | Expected Viol. | Actual Viol. | Kupiec | Christoffersen | Basel |
|---|---|---|---|---|---|---|
| 95% | HS | 37.65 | 38 | Pass | Pass | — |
| 95% | Normal | 37.65 | 34 | Pass | Pass | — |
| 95% | Student-t | 37.65 | 34 | Pass | Pass | — |
| 99% | HS | 7.53 | 13 | Pass | **Fail** | 🔴 Red |
| 99% | Normal | 7.53 | 14 | **Fail** | **Fail** | 🔴 Red |
| 99% | Student-t | 7.53 | 13 | Pass | **Fail** | 🔴 Red |

- All models fall into Basel red zone at 99% — violations cluster in 2022–2023
  and 2024–2025 volatile periods, exposing the i.i.d. returns assumption
- Recommended model: Historical Simulation — exact violations at 95%, passes
  Kupiec at 99%, avoids distributional misspecification

**Exercise IV — Multivariate Distribution Analysis**
- Jarque-Bera tests reject normality for all 5 stocks at all significance levels
- Student-t degrees of freedom: INTC (3.41) has fattest tail; NVDA (5.26) thinnest
- Mardia's multivariate normality test (implemented from scratch): both skewness
  and kurtosis components reject joint normality (p ≈ 0.0)
- Normal assumption underestimates 99% VaR by ~13% vs empirical distribution
- Diversification benefit: ~20% reduction in VaR across all methods and confidence
  levels (portfolio VaR consistently ~20% below sum of individual VaRs)

## Key results

| Method | VaR₀.₉₅ | VaR₀.₉₉ | Diversification benefit (99%) |
|---|---|---|---|
| Multivariate Normal | 3.25% | 4.62% | 20.85% |
| Multivariate Student-t | 3.09% | 5.05% | 23.57% |
| Empirical | 3.23% | 5.32% | 19.66% |

## Tech stack

Python · NumPy · SciPy (`multivariate_t`, `norm`, `t`, `chi2`, `jarque_bera`) ·
statsmodels (OLS) · scikit-learn (PCA) · Matplotlib · Pandas · yFinance ·
`arch` library


## Data

5 years of daily prices for AAPL, MSFT, NVDA, AMD, INTC, NASDAQ (^IXIC),
S&P 500 (^GSPC). Source: Yahoo Finance via `yfinance`.
Note: live data is pulled on execution — results may differ slightly from the
report, which used data collected prior to September 7, 2025.
