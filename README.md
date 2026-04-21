# Quantitative-Risk-Management

Coursework repository Quantitative Risk Management course. Three coursework projects covering the core toolkit of modern quantitative
risk: factor models, VaR estimation and backtesting, volatility modelling, extreme
value theory, and copula-based dependence modelling. All implemented in Python.

## coursework projects

### [Coursework project 1 — Factor Models, PCA & VaR Backtesting](./Factor_PCA_VaR/)
Two-factor market model for a 5-stock tech portfolio (AAPL, MSFT, NVDA, AMD, INTC),
PCA-based risk decomposition, and comprehensive VaR backtesting across three methods
and two confidence levels.

**Highlights**
- NVDA and AMD carry ~3× NASDAQ exposure vs other stocks; INTC is 70% idiosyncratic
- PC1 alone drives 51% of portfolio VaR — identified as primary hedging target
- All models fall into Basel red zone at 99% VaR due to volatility clustering in 2022–2023
- Mardia's multivariate normality test implemented from scratch; normality rejected for
  all stocks

### [Coursework project 2 — Volatility Modelling & Market Crash Prediction](./Volatility_Modeling/)
Replication of Kim et al. (2011) using S&P 500 daily returns (2001–2020). Six volatility
models estimated and evaluated on their ability to predict the March 2020 COVID crash.

**Highlights**
- CV (Normal) estimates crash probability at 2.62 × 10⁻²³ — effectively impossible
- GARCH (Student-t) gives shortest recurrence: ~25 trading days — closest to reality
- ARMA(1,1)-GARCH(1,1) implemented manually via custom MLE — no Python package
  supports joint estimation; used `scipy.optimize.minimize` with L-BFGS-B
- Only GARCH (Student-t) passes all Christoffersen backtesting tests

### [Coursework project 3 — Extreme Value Theory & Aggregate Risk](./ExtremeValueTheory&AggRisk/)
EVT analysis of Nikkei 225 tail risk (2015–2025) using block maxima (GEV) and
peaks-over-threshold (GPD/POT), plus copula-based aggregate risk for a 3-asset portfolio.

**Highlights**
- GEV fit improves substantially with block size: ξ drops from 0.77 (n=10) to 0.32 (n=40)
- POT model: GPD shape ξ = 0.270 closely matches GEV at n=40 (ξ = 0.323),
  confirming the Pickands–Balkema–de Haan theorem empirically
- Kupiec POF test on 20% holdout: 7 violations vs 4.89 expected, p = 0.37 — model adequate
- Student-t copula (ρ=0.5, ν=4) increases portfolio VaR₀.₉₉ by 38% vs independence,
  demonstrating the material impact of tail dependence on capital requirements

## Skills demonstrated

| Area | Methods |
|---|---|
| Market risk modelling | Historical simulation, parametric VaR, rolling window forecasting |
| Volatility modelling | CV, GARCH(1,1), ARMA-GARCH — Normal and Student-t innovations |
| Extreme value theory | GEV (block maxima), GPD (POT), return level estimation |
| Dependence modelling | Gaussian copula, Student-t copula, comonotonic bounds |
| Risk attribution | PCA decomposition, factor model, systematic vs idiosyncratic risk |
| Backtesting | Kupiec POF, Christoffersen independence, Basel traffic light |
| Statistical testing | KS test, Jarque-Bera, Mardia multivariate normality |
| Implementation | Custom MLE, bootstrap standard errors, Monte Carlo simulation |


## Tech stack

Python · NumPy · SciPy · Pandas · Matplotlib · statsmodels · scikit-learn ·
`arch` · yFinance 

## Reference

Kim, Y.S., Rachev, S.T., Bianchi, M.L., Mitov, I. and Fabozzi, F.J. (2011).
Time series analysis for financial market meltdowns.
*Journal of Banking & Finance*, 35, 1879–1891.
