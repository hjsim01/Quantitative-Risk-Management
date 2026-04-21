import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import norm,t,kstest,chi2
from scipy.optimize import minimize
from arch import arch_model
import matplotlib.pyplot as plt

# Download data from Jan 10, 2000 to one day before Covid crash
start_date = "2000-01-10"
end_date = "2020-03-15"  

data = yf.download("^GSPC", start=start_date, end=end_date, interval="1d")
close_prices = data['Close']

# Calculate log returns 
#(*100 so that the return in %, and the warning will not show up)
returns = 100 * np.log(close_prices / close_prices.shift(1)).dropna()

# Convert to numpy array
r = returns.values

plt.figure(figsize=(12, 6))
plt.plot(returns.index, returns.values, linewidth=1)
plt.title('Returns Time Series', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Returns')
plt.grid(True, alpha=0.3)
plt.show()

# CV model with normal innovations
cv_normal = arch_model(r, vol='Constant', dist='normal', mean='constant')
res_cv_normal = cv_normal.fit(disp="off")
print("CV Normal:\n", res_cv_normal.summary())
# CV model with Student-t innovations
cv_t = arch_model(r, vol='Constant', dist='t', mean='constant')
res_cv_t = cv_t.fit(disp="off")
print("CV Student-t:\n", res_cv_t.summary())

# GARCH(1,1) with normal innovations
garch_normal = arch_model(r, mean='constant',vol='GARCH', p=1, q=1, dist='normal')
res_garch_normal = garch_normal.fit(disp="off")
print("GARCH Normal:\n", res_garch_normal.summary())

# GARCH(1,1) with Student-t innovations
garch_t = arch_model(r, vol='GARCH', p=1, q=1, dist='t', mean='constant')
res_garch_t = garch_t.fit(disp="off")
print("GARCH Student-t:\n", res_garch_t.summary())

# Manual ARMA(1,1)-GARCH(1,1) with normal innovations
def log_likelihood_arma_garch(params, r):
    mu, phi, theta, omega, alpha, beta = params
    T = len(r)
    
    eps = np.zeros(T)
    sigma2 = np.zeros(T)
    
    eps[0] = r[0] - mu
    sigma2[0] = np.var(r)
    
    for t in range(1, T):
        eps[t] = r[t] - mu - phi * r[t - 1] - theta * eps[t - 1]
        sigma2[t] = omega + alpha * eps[t - 1]**2 + beta * sigma2[t - 1]

    cond_std = np.sqrt(sigma2)
    log_pdf = norm.logpdf(eps, loc=0, scale=cond_std)
    loglik = np.sum(log_pdf)
    
    return -loglik  # for minimization

# Initial guesses: [mu, phi, theta, omega, alpha, beta]
init_params = [0, 0.1, 0.1, 0.01, 0.05, 0.9]

# Define parameter bounds
bounds = [
    (None, None),   # mu
    (-0.99, 0.99),  # phi
    (-0.99, 0.99),  # theta
    (1e-6, None),   # omega
    (0, 1),         # alpha
    (0, 1)          # beta
]

result = minimize(
    fun=lambda p: log_likelihood_arma_garch(p, r),
    x0=init_params,
    bounds=bounds,
    method='L-BFGS-B'
)

# Put paramter a list (prepare for dictionary)
param_names = ["c", "a", "b", "alpha0", "alpha1", "beta1"]
estimated_params = result.x

param_df = pd.DataFrame({
    'Parameter': param_names,
    'Estimate': estimated_params
})
print(param_df)

# Manual ARMA(1,1)-GARCH(1,1) with Student-t innovations
#ARMA-GARCH with Student-t
def log_likelihood_arma_garch_t(params, r):
    mu, phi, theta, omega, alpha, beta, nu = params
    T = len(r)
    
    eps = np.zeros(T)
    sigma2 = np.zeros(T)
    
    eps[0] = r[0] - mu
    sigma2[0] = np.var(r)

    for i in range(1, T):  
        eps[i] = r[i] - mu - phi * r[i - 1] - theta * eps[i - 1]
        sigma2[i] = omega + alpha * eps[i - 1]**2 + beta * sigma2[i - 1]
    
    cond_std = np.sqrt(sigma2)
    z = eps / cond_std

    # ✅ Now this works correctly
    logpdf = t.logpdf(z, df=nu) - np.log(cond_std)

    loglik = np.sum(logpdf)
    return -loglik


# Initial guess: [mu, phi, theta, omega, alpha, beta, nu]
init_params_t = [0, 0.1, 0.1, 0.01, 0.05, 0.9, 8]  # nu starts at 8

# Bounds (to keep model valid)
bounds_t = [
    (None, None),     # mu
    (-0.99, 0.99),    # phi
    (-0.99, 0.99),    # theta
    (1e-6, None),     # omega
    (0, 1),           # alpha
    (0, 1),           # beta
    (2.1, 100)        # nu: must be >2 to ensure finite variance
]

# Minimize the negative log-likelihood
result_t = minimize(
    fun=lambda p: log_likelihood_arma_garch_t(p, r),
    x0=init_params_t,
    bounds=bounds_t,
    method='L-BFGS-B'
)
# Put paramter a list (prepare for dictionary)
param_names_t = ["c", "a", "b", "alpha0", "alpha1", "beta1","df"]
estimated_params_t = result_t.x

param_df_t = pd.DataFrame({
    'Parameter': param_names_t,
    'Estimate': estimated_params_t
})
print(param_df_t)

# ============================================
# CRASH PROBABILITY ANALYSIS - ALL 6 MODELS
# ============================================

crash_return = -12  # Define crash scenario
results_table = []

def get_standardized_residuals_arma_garch(params, r, is_studentt=False):
    """Calculate standardized residuals for ARMA-GARCH"""
    if is_studentt:
        mu, phi, theta, omega, alpha, beta, nu = params
    else:
        mu, phi, theta, omega, alpha, beta = params
    
    T = len(r)
    eps = np.zeros(T)
    sigma2 = np.zeros(T)
    
    eps[0] = r[0] - mu
    sigma2[0] = np.var(r)
    
    for t in range(1, T):
        eps[t] = r[t] - mu - phi * r[t-1] - theta * eps[t-1]
        sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
        sigma2[t] = max(sigma2[t], 1e-6)  # Prevent negative variance
    
    z = eps / np.sqrt(sigma2)
    return z, sigma2, eps

def forecast_arma_garch(params, r, is_studentt=False):
    """One-step ahead forecast for ARMA-GARCH"""
    z, sigma2, eps = get_standardized_residuals_arma_garch(params, r, is_studentt)
    
    if is_studentt:
        mu, phi, theta, omega, alpha, beta, nu = params
    else:
        mu, phi, theta, omega, alpha, beta = params
        nu = None
    
    r_T = r[-1]
    eps_T = eps[-1]
    sigma2_T = sigma2[-1]
    
    mu_forecast = mu + phi * r_T + theta * eps_T
    sigma2_forecast = omega + alpha * eps_T**2 + beta * sigma2_T
    sigma_forecast = np.sqrt(max(sigma2_forecast, 1e-6))
    
    return mu_forecast, sigma_forecast, nu

# ============================================
# 1. CV NORMAL
# ============================================
mu_cv = res_cv_normal.params['mu']
sigma2_cv = res_cv_normal.params['sigma2']
sigma_cv = np.sqrt(sigma2_cv)

# Last residual
last_return = r[-1]
residual_cv_normal = (last_return - mu_cv) / sigma_cv

# Probability
prob_cv_normal = norm.cdf(crash_return, loc=mu_cv, scale=sigma_cv)
avg_time_cv_normal = 1 / (prob_cv_normal * 250)

results_table.append({
    'Model': 'CV',
    'Innovation': 'Normal',
    'Residual': residual_cv_normal,
    'Probability': prob_cv_normal,
    'Avg_Time_Years': avg_time_cv_normal
})

# ============================================
# 2. CV STUDENT-T
# ============================================
mu_cv_t = res_cv_t.params['mu']
sigma2_cv_t = res_cv_t.params['sigma2']
sigma_cv_t = np.sqrt(sigma2_cv_t)
nu_cv_t = res_cv_t.params['nu']

# Last residual
residual_cv_t = (last_return - mu_cv_t) / sigma_cv_t

# Probability
z_crash_cv = (crash_return - mu_cv_t) / sigma_cv_t
prob_cv_t = t.cdf(z_crash_cv, df=nu_cv_t)
avg_time_cv_t = 1 / (prob_cv_t * 250)

results_table.append({
    'Model': 'CV',
    'Innovation': 'Student-t',
    'Residual': residual_cv_t,
    'Probability': prob_cv_t,
    'Avg_Time_Years': avg_time_cv_t
})

# ============================================
# 3. GARCH NORMAL
# ============================================
mu_g = res_garch_normal.params['mu']
fitted_g = res_garch_normal.conditional_volatility
sigma_g_last = fitted_g[-1]  # Use array indexing, not .iloc

# Last residual
residual_garch_normal = (last_return - mu_g) / sigma_g_last

# Forecast next period volatility
omega_g = res_garch_normal.params['omega']
alpha_g = res_garch_normal.params['alpha[1]']
beta_g = res_garch_normal.params['beta[1]']

eps_last = last_return - mu_g
sigma2_forecast_g = omega_g + alpha_g * eps_last**2 + beta_g * sigma_g_last**2
sigma_forecast_g = np.sqrt(sigma2_forecast_g)

# Probability
prob_garch_normal = norm.cdf(crash_return, loc=mu_g, scale=sigma_forecast_g)
avg_time_garch_normal = 1 / (prob_garch_normal * 250)

results_table.append({
    'Model': 'GARCH',
    'Innovation': 'Normal',
    'Residual': residual_garch_normal,
    'Probability': prob_garch_normal,
    'Avg_Time_Years': avg_time_garch_normal
})

# ============================================
# 4. GARCH STUDENT-T
# ============================================
mu_gt = res_garch_t.params['mu']
fitted_gt = res_garch_t.conditional_volatility
sigma_gt_last = fitted_gt[-1]  # Use array indexing, not .iloc
nu_gt = res_garch_t.params['nu']

# Last residual
residual_garch_t = (last_return - mu_gt) / sigma_gt_last

# Forecast
omega_gt = res_garch_t.params['omega']
alpha_gt = res_garch_t.params['alpha[1]']
beta_gt = res_garch_t.params['beta[1]']

eps_last_gt = last_return - mu_gt
sigma2_forecast_gt = omega_gt + alpha_gt * eps_last_gt**2 + beta_gt * sigma_gt_last**2
sigma_forecast_gt = np.sqrt(sigma2_forecast_gt)

# Probability
z_crash_gt = (crash_return - mu_gt) / sigma_forecast_gt
prob_garch_t = t.cdf(z_crash_gt, df=nu_gt)
avg_time_garch_t = 1 / (prob_garch_t * 250)

results_table.append({
    'Model': 'GARCH',
    'Innovation': 'Student-t',
    'Residual': residual_garch_t,
    'Probability': prob_garch_t,
    'Avg_Time_Years': avg_time_garch_t
})

# ============================================
# 5. ARMA-GARCH NORMAL
# ============================================
params_arma_normal = result.x  # From your manual estimation

z_arma_n, sigma2_arma_n, eps_arma_n = get_standardized_residuals_arma_garch(
    params_arma_normal, r, is_studentt=False
)
residual_arma_normal = z_arma_n[-1]

mu_f_arma_n, sigma_f_arma_n, _ = forecast_arma_garch(
    params_arma_normal, r, is_studentt=False
)

prob_arma_normal = norm.cdf(crash_return, loc=mu_f_arma_n, scale=sigma_f_arma_n)
avg_time_arma_normal = 1 / (prob_arma_normal * 250)

results_table.append({
    'Model': 'ARMA-GARCH',
    'Innovation': 'Normal',
    'Residual': residual_arma_normal,
    'Probability': prob_arma_normal,
    'Avg_Time_Years': avg_time_arma_normal
})

# ============================================
# 6. ARMA-GARCH STUDENT-T
# ============================================
params_arma_t = result_t.x  # From your manual estimation

z_arma_t, sigma2_arma_t, eps_arma_t = get_standardized_residuals_arma_garch(
    params_arma_t, r, is_studentt=True
)
residual_arma_t = z_arma_t[-1]

mu_f_arma_t, sigma_f_arma_t, nu_arma_t = forecast_arma_garch(
    params_arma_t, r, is_studentt=True
)

z_crash_arma = (crash_return - mu_f_arma_t) / sigma_f_arma_t
prob_arma_t = t.cdf(z_crash_arma, df=nu_arma_t)
avg_time_arma_t = 1 / (prob_arma_t * 250)

results_table.append({
    'Model': 'ARMA-GARCH',
    'Innovation': 'Student-t',
    'Residual': residual_arma_t,
    'Probability': prob_arma_t,
    'Avg_Time_Years': avg_time_arma_t
})

# ============================================
# DISPLAY RESULTS
# ============================================
df_results = pd.DataFrame(results_table)

print("\n" + "="*80)
print(f"CRASH PROBABILITY ANALYSIS: {crash_return}% Daily Return")
print( "="*80)

for _, row in df_results.iterrows():
    print(f"{row['Model']:12} {row['Innovation']:10} | "
        f"Residual: {float(row['Residual']):6.4f} | "
        f"Prob: {float(row['Probability']):.2e} | "
        f"Avg Time: {float(row['Avg_Time_Years']):.4f} years")

# ============================================
# GOODNESS-OF-FIT FUNCTIONS
# ============================================

def goodness_of_fit_tests(residuals, distribution='norm', df=None):
    """Calculate KS statistics only"""
    
    # KS Test
    if distribution == 'norm':
        ks_stat, ks_pvalue = kstest(residuals, 'norm', args=(0, 1))
    elif distribution == 't':
        ks_stat, ks_pvalue = kstest(residuals, 't', args=(df, 0, 1))
    
    return ks_stat, ks_pvalue

# ============================================
# CALCULATE FOR ALL 6 MODELS
# ============================================

gof_results = []

# 1. CV Normal
z_cv_normal = (r - mu_cv) / sigma_cv
ks, ksp = goodness_of_fit_tests(z_cv_normal, 'norm')
gof_results.append({'Model': 'CV', 'Innovation': 'Normal', 'KS_stat': ks, 'KS_pvalue': ksp})

# 2. CV Student-t
z_cv_t = (r - mu_cv_t) / sigma_cv_t
ks, ksp = goodness_of_fit_tests(z_cv_t, 't', df=nu_cv_t)
gof_results.append({'Model': 'CV', 'Innovation': 'Student-t', 'KS_stat': ks, 'KS_pvalue': ksp})

# 3. GARCH Normal
z_garch_normal = res_garch_normal.std_resid
ks, ksp = goodness_of_fit_tests(z_garch_normal, 'norm')
gof_results.append({'Model': 'GARCH', 'Innovation': 'Normal', 'KS_stat': ks, 'KS_pvalue': ksp})

# 4. GARCH Student-t
z_garch_t = res_garch_t.std_resid
ks, ksp = goodness_of_fit_tests(z_garch_t, 't', df=nu_gt)
gof_results.append({'Model': 'GARCH', 'Innovation': 'Student-t', 'KS_stat': ks, 'KS_pvalue': ksp})

# 5. ARMA-GARCH Normal
z_arma_normal, _, _ = get_standardized_residuals_arma_garch(params_arma_normal, r, is_studentt=False)
ks, ksp = goodness_of_fit_tests(z_arma_normal, 'norm')
gof_results.append({'Model': 'ARMA-GARCH', 'Innovation': 'Normal', 'KS_stat': ks, 'KS_pvalue': ksp})

# 6. ARMA-GARCH Student-t
z_arma_t, _, _ = get_standardized_residuals_arma_garch(params_arma_t, r, is_studentt=True)
nu_arma = params_arma_t[6]
ks, ksp = goodness_of_fit_tests(z_arma_t, 't', df=nu_arma)
gof_results.append({'Model': 'ARMA-GARCH', 'Innovation': 'Student-t', 'KS_stat': ks, 'KS_pvalue': ksp})

# ============================================
# DISPLAY RESULTS
# ============================================

df_gof = pd.DataFrame(gof_results)

print("\n" + "="*80)
print("GOODNESS-OF-FIT TESTS (KS Statistics)")
print("="*80)
print(df_gof.to_string(index=False))

print("\n" + "="*80)
print("INTERPRETATION (at 1% significance level)")
print("="*80)
for _, row in df_gof.iterrows():
    reject_ks = "REJECT" if float(row['KS_pvalue']) < 0.01 else "ACCEPT"
    print(f"{row['Model']:12} {row['Innovation']:10} | "
          f"KS: {float(row['KS_stat']):.4f} (p={float(row['KS_pvalue']):.4f}) [{reject_ks}]")

print("- Models are REJECTED if KS p-value < 0.01")

# Function for ES
def calculate_expected_shortfall(mu, sigma, alpha, distribution='norm', df=None):
 
    if distribution == 'norm':
        # VaR
        var = norm.ppf(alpha, loc=mu, scale=sigma)
        
        # ES = μ + σ * φ(Φ^(-1)(α)) / α
        z_alpha = norm.ppf(alpha)
        phi_z = norm.pdf(z_alpha)
        es = mu + sigma * phi_z / alpha
        
    elif distribution == 't':
        # VaR
        var = -mu + sigma * t.ppf(alpha, df=df)
        
        # ES for Student-t
        t_alpha = t.ppf(alpha, df=df)
        f_t = t.pdf(t_alpha, df=df)
        es = mu + sigma * f_t / alpha * (df + t_alpha**2) / (df - 1)
    
    return var, es

# ============================================
# Calculate ES for all 6 models
# ============================================

es_results = []

# 1. CV Normal
var_cv_n, es_cv_n = calculate_expected_shortfall(mu_cv, sigma_cv, 0.01, 'norm')
es_results.append({'Model': 'CV', 'Innovation': 'Normal', 'VaR_99': var_cv_n, 'ES_99': es_cv_n})

# 2. CV Student-t
var_cv_t, es_cv_t = calculate_expected_shortfall(mu_cv_t, sigma_cv_t, 0.01, 't', df=nu_cv_t)
es_results.append({'Model': 'CV', 'Innovation': 'Student-t', 'VaR_99': var_cv_t, 'ES_99': es_cv_t})

# 3. GARCH Normal
var_g_n, es_g_n = calculate_expected_shortfall(mu_g, sigma_forecast_g, 0.01, 'norm')
es_results.append({'Model': 'GARCH', 'Innovation': 'Normal', 'VaR_99': var_g_n, 'ES_99': es_g_n})

# 4. GARCH Student-t
var_g_t, es_g_t = calculate_expected_shortfall(mu_gt, sigma_forecast_gt, 0.01, 't', df=nu_gt)
es_results.append({'Model': 'GARCH', 'Innovation': 'Student-t', 'VaR_99': var_g_t, 'ES_99': es_g_t})

# 5. ARMA-GARCH Normal
var_ag_n, es_ag_n = calculate_expected_shortfall(mu_f_arma_n, sigma_f_arma_n, 0.01, 'norm')
es_results.append({'Model': 'ARMA-GARCH', 'Innovation': 'Normal', 'VaR_99': var_ag_n, 'ES_99': es_ag_n})

# 6. ARMA-GARCH Student-t
var_ag_t, es_ag_t = calculate_expected_shortfall(mu_f_arma_t, sigma_f_arma_t, 0.01, 't', df=nu_arma_t)
es_results.append({'Model': 'ARMA-GARCH', 'Innovation': 'Student-t', 'VaR_99': var_ag_t, 'ES_99': es_ag_t})

# ============================================
# DISPLAY RESULTS
# ============================================

df_es = pd.DataFrame(es_results)

print("\n" + "="*80)
print("VaR AND EXPECTED SHORTFALL (99% Confidence Level)")
print("="*80)
print(df_es.to_string(index=False))

# ============================================
# ROLLING WINDOW: VaR AND ES FOR 4 MODELS
# ============================================

window_size = len(r) - 2500
forecast_horizon = 2500  # Forecast the last 250 days

# Storage dictionaries for all 4 models
forecasts = {
    'cv_normal': {'var': [], 'es': [], 'mu': [], 'sigma': []},
    'cv_t': {'var': [], 'es': [], 'mu': [], 'sigma': [], 'nu': []},
    'garch_normal': {'var': [], 'es': [], 'mu': [], 'sigma': []},
    'garch_t': {'var': [], 'es': [], 'mu': [], 'sigma': [], 'nu': []}
}

print(f"Rolling window size: {window_size} days")
print(f"Forecast horizon: {forecast_horizon} days")
print("Estimating models and calculating VaR & ES...")

# Rolling window forecasts
for i in range(forecast_horizon):
    if i % 50 == 0:
        print(f"Progress: {i}/{forecast_horizon}")
    
    # Subset data
    r_subset = r[i:window_size + i]
    
    # ========================================
    # 1. CV NORMAL
    # ========================================
    cv_normal_temp = arch_model(r_subset, vol='Constant', dist='normal', mean='constant')
    res_cv_normal = cv_normal_temp.fit(disp='off')
    
    mu_cv_n = res_cv_normal.params['mu']
    sigma_cv_n = np.sqrt(res_cv_normal.params['sigma2'])
    
    # VaR
    var_cv_n = norm.ppf(0.01, loc=mu_cv_n, scale=sigma_cv_n)
    
    # ES (Expected Shortfall)
    z_alpha = norm.ppf(0.01)
    phi_z = norm.pdf(z_alpha)
    es_cv_n = mu_cv_n + sigma_cv_n * phi_z / 0.01
    
    forecasts['cv_normal']['var'].append(var_cv_n)
    forecasts['cv_normal']['es'].append(es_cv_n)
    forecasts['cv_normal']['mu'].append(mu_cv_n)
    forecasts['cv_normal']['sigma'].append(sigma_cv_n)
    
    # ========================================
    # 2. CV STUDENT-T
    # ========================================
    cv_t_temp = arch_model(r_subset, vol='Constant', dist='t', mean='constant')
    res_cv_t = cv_t_temp.fit(disp='off')
    
    mu_cv_t = res_cv_t.params['mu']
    sigma_cv_t = np.sqrt(res_cv_t.params['sigma2'])
    nu_cv_t = res_cv_t.params['nu']
    
    # VaR
    var_cv_t = mu_cv_t + sigma_cv_t * t.ppf(0.01, df=nu_cv_t)
    
    # ES (Expected Shortfall)
    t_alpha = t.ppf(0.01, df=nu_cv_t)
    f_t = t.pdf(t_alpha, df=nu_cv_t)
    es_cv_t = mu_cv_t + sigma_cv_t * f_t / 0.01 * (nu_cv_t + t_alpha**2) / (nu_cv_t - 1)
    
    forecasts['cv_t']['var'].append(var_cv_t)
    forecasts['cv_t']['es'].append(es_cv_t)
    forecasts['cv_t']['mu'].append(mu_cv_t)
    forecasts['cv_t']['sigma'].append(sigma_cv_t)
    forecasts['cv_t']['nu'].append(nu_cv_t)
    
    # ========================================
    # 3. GARCH NORMAL
    # ========================================
    garch_normal_temp = arch_model(r_subset, vol='GARCH', p=1, q=1, dist='normal', mean='constant')
    res_garch_normal = garch_normal_temp.fit(disp='off')
    
    # Forecast
    forecast_gn = res_garch_normal.forecast(horizon=1)
    mu_gn = res_garch_normal.params['mu']
    sigma_gn = np.sqrt(forecast_gn.variance.values[-1, 0])
    
    # VaR
    var_gn = norm.ppf(0.01, loc=mu_gn, scale=sigma_gn)
    
    # ES
    es_gn = mu_gn + sigma_gn * phi_z / 0.01
    
    forecasts['garch_normal']['var'].append(var_gn)
    forecasts['garch_normal']['es'].append(es_gn)
    forecasts['garch_normal']['mu'].append(mu_gn)
    forecasts['garch_normal']['sigma'].append(sigma_gn)
    
    # ========================================
    # 4. GARCH STUDENT-T
    # ========================================
    garch_t_temp = arch_model(r_subset, vol='GARCH', p=1, q=1, dist='t', mean='constant')
    res_garch_t = garch_t_temp.fit(disp='off')
    
    # Forecast
    forecast_gt = res_garch_t.forecast(horizon=1)
    mu_gt = res_garch_t.params['mu']
    sigma_gt = np.sqrt(forecast_gt.variance.values[-1, 0])
    nu_gt = res_garch_t.params['nu']
    
    # VaR
    var_gt = mu_gt + sigma_gt * t.ppf(0.01, df=nu_gt)
    
    # ES
    t_alpha_gt = t.ppf(0.01, df=nu_gt)
    f_t_gt = t.pdf(t_alpha_gt, df=nu_gt)
    es_gt = mu_gt + sigma_gt * f_t_gt / 0.01 * (nu_gt + t_alpha_gt**2) / (nu_gt - 1)
    
    forecasts['garch_t']['var'].append(var_gt)
    forecasts['garch_t']['es'].append(es_gt)
    forecasts['garch_t']['mu'].append(mu_gt)
    forecasts['garch_t']['sigma'].append(sigma_gt)
    forecasts['garch_t']['nu'].append(nu_gt)



# Convert all to arrays
for model in forecasts:
    for key in forecasts[model]:
        forecasts[model][key] = np.array(forecasts[model][key])

# Actual returns for comparison
actual_returns = r[window_size:]

# ============================================
# DISPLAY VAR AND ES STATISTICS
# ============================================

print("\n" + "="*80)
print("VaR AND ES SUMMARY STATISTICS")
print("="*80)

models = ['cv_normal', 'cv_t', 'garch_normal', 'garch_t']
model_names = {
    'cv_normal': 'CV Normal',
    'cv_t': 'CV Student-t',
    'garch_normal': 'GARCH Normal',
    'garch_t': 'GARCH Student-t'
}

summary_data = []
for model_key in models:
    var_mean = forecasts[model_key]['var'].mean()
    var_std = forecasts[model_key]['var'].std()
    var_min = forecasts[model_key]['var'].min()
    var_max = forecasts[model_key]['var'].max()
    
    es_mean = forecasts[model_key]['es'].mean()
    es_std = forecasts[model_key]['es'].std()
    es_min = forecasts[model_key]['es'].min()
    es_max = forecasts[model_key]['es'].max()
    
    summary_data.append({
        'Model': model_names[model_key],
        'VaR_Mean': var_mean,
        'VaR_Std': var_std,
        'VaR_Min': var_min,
        'VaR_Max': var_max,
        'ES_Mean': es_mean,
        'ES_Std': es_std,
        'ES_Min': es_min,
        'ES_Max': es_max
    })

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

# ============================================
# VIOLATIONS CHECK
# ============================================

print("\n" + "="*80)
print("VaR VIOLATIONS (Preliminary Check)")
print("="*80)

for model_key in models:
    violations = (actual_returns < forecasts[model_key]['var']).sum()
    violation_rate = violations / len(actual_returns)
    
    print(f"\n{model_names[model_key]}:")
    print(f"  Violations: {violations} out of {len(actual_returns)} (expected: 2.5)")
    print(f"  Violation rate: {violation_rate:.4f} (expected: 0.01)")

# ============================================
# CLR and BLR Test
# ============================================

def christoffersen_test(returns, var_forecasts, alpha=0.01):
    """
    Christoffersen LR test - FINAL CORRECTED VERSION
    """
    returns = np.array(returns).flatten()
    var_forecasts = np.array(var_forecasts).flatten()
    
    n = len(returns)
    violations = (returns < var_forecasts).astype(int)
    n_violations = int(violations.sum())
    expected_violations = n * alpha
    
    # 1. Unconditional Coverage Test
    if n_violations == 0:
        lr_uc = 2 * n * np.log(1 - alpha)
    elif n_violations == n:
        lr_uc = 2 * n * np.log(alpha)
    else:
        p_hat = n_violations / n
        lr_uc = 2 * (
            n_violations * (np.log(p_hat) - np.log(alpha)) + 
            (n - n_violations) * (np.log(1 - p_hat) - np.log(1 - alpha))
        )
    p_value_uc = 1 - chi2.cdf(lr_uc, df=1)
    
    # 2. Independence Test - FINAL FIX
    n00 = n01 = n10 = n11 = 0
    for i in range(len(violations) - 1):
        if violations[i] == 0 and violations[i+1] == 0:
            n00 += 1
        elif violations[i] == 0 and violations[i+1] == 1:
            n01 += 1
        elif violations[i] == 1 and violations[i+1] == 0:
            n10 += 1
        elif violations[i] == 1 and violations[i+1] == 1:
            n11 += 1
    
    n_0 = n00 + n01
    n_1 = n10 + n11
    
    # Special case: if n11=0 (no consecutive violations), we still need to test
    if n_violations < 2 or n_1 == 0:
        # When violations are perfectly spread out (no clustering)
        # This is actually GOOD for independence
        lr_ind = 0
        p_value_ind = 1.0
    else:
        # Calculate probabilities with small constant to avoid log(0)
        epsilon = 1e-10
        p01 = (n01 + epsilon) / (n_0 + epsilon) if n_0 > 0 else epsilon
        p11 = (n11 + epsilon) / (n_1 + epsilon) if n_1 > 0 else epsilon
        p = (n_violations + epsilon) / (n + epsilon)
        
        # Calculate likelihood ratio
        # Under H1 (conditional): separate probabilities for state 0 and 1
        # Under H0 (unconditional): same probability regardless of previous state
        logL1 = (n00 * np.log(1 - p01) + n01 * np.log(p01) + 
                 n10 * np.log(1 - p11) + n11 * np.log(p11))
        logL0 = ((n00 + n10) * np.log(1 - p) + (n01 + n11) * np.log(p))
        
        lr_ind = -2 * (logL0 - logL1)
        lr_ind = max(0, lr_ind)  # Ensure non-negative
        p_value_ind = 1 - chi2.cdf(lr_ind, df=1)
    
    # 3. Conditional Coverage
    lr_cc = lr_uc + lr_ind
    p_value_cc = 1 - chi2.cdf(lr_cc, df=2)
    
    return {
        'n_observations': n,
        'n_violations': n_violations,
        'expected_violations': expected_violations,
        'violation_rate': n_violations / n,
        'transitions': {'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11},
        'LR_uc': lr_uc,
        'p_value_uc': p_value_uc,
        'LR_ind': lr_ind,
        'p_value_ind': p_value_ind,
        'LR_cc': lr_cc,
        'p_value_cc': p_value_cc
    }

# ============================================
# FINAL CORRECTED BERKOWITZ TEST
# ============================================

def berkowitz_test(returns, mu_forecasts, sigma_forecasts, alpha=0.01, 
                   distribution='norm', df=None):
    """
    Berkowitz test - FINAL CORRECTED VERSION
    """
    returns = np.array(returns).flatten()
    mu_forecasts = np.array(mu_forecasts).flatten()
    sigma_forecasts = np.array(sigma_forecasts).flatten()
    
    n = len(returns)
    
    # Transform to uniform [0,1]
    if distribution == 'norm':
        u = norm.cdf(returns, loc=mu_forecasts, scale=sigma_forecasts)
    elif distribution == 't':
        z_raw = (returns - mu_forecasts) / sigma_forecasts
        u = t.cdf(z_raw, df=df)
    
    # Clip to avoid extremes
    u = np.clip(u, 1e-10, 1 - 1e-10)
    
    # Transform to standard normal
    z = norm.ppf(u)
    
    # Remove invalid
    valid = np.isfinite(z)
    z = z[valid]
    n_valid = len(z)
    
    if n_valid < 10:
        return {
            'n_observations': n,
            'z_mean': np.nan,
            'z_var': np.nan,
            'LR_tail': np.nan,
            'p_value_tail': np.nan,
            'LR_ind': np.nan,
            'p_value_ind': np.nan
        }
    
    z_mean = z.mean()
    z_var = z.var(ddof=1)
    
    # Test 1: Joint test for mean=0 and variance=1
    # Correct formula: LR = n * [mean^2/var + log(var) + 1 - var]
    # But we need to be careful with the formula
    
    # Simpler approach: separate tests
    # Test if mean = 0
    lr_mean = n_valid * (z_mean**2 / z_var)
    
    # Test if variance = 1
    # LR for variance test: n * [(s^2 - 1)^2 / (2*s^2)]
    # Or use: n * [log(s^2) - (s^2 - 1)]
    if z_var > 0:
        lr_var = n_valid * (np.log(z_var) - (z_var - 1))
        lr_var = abs(lr_var)  # Take absolute value
    else:
        lr_var = 0
    
    # Combined tail test
    lr_tail = lr_mean + lr_var
    p_value_tail = 1 - chi2.cdf(lr_tail, df=2)
    
    # Test 2: Independence (AR(1))
    if n_valid > 2:
        # Standardize z
        z_standardized = (z - z_mean) / np.sqrt(z_var)
        
        # Calculate sample autocorrelation at lag 1
        z_t = z_standardized[:-1]
        z_t1 = z_standardized[1:]
        
        rho1 = np.corrcoef(z_t, z_t1)[0, 1]
        
        # LR test: n * rho^2 ~ chi2(1) under H0: rho=0
        lr_ind = (n_valid - 1) * rho1**2
        p_value_ind = 1 - chi2.cdf(lr_ind, df=1)
    else:
        lr_ind = 0
        p_value_ind = 1.0
    
    return {
        'n_observations': n,
        'n_valid': n_valid,
        'z_mean': z_mean,
        'z_var': z_var,
        'LR_tail': lr_tail,
        'p_value_tail': p_value_tail,
        'LR_ind': lr_ind,
        'p_value_ind': p_value_ind
    }

# ============================================
# RUN FINAL CORRECTED TESTS
# ============================================

print("\n" + "="*80)
print("CHRISTOFFERSEN AND BERKOWITZ TEST RESULTS (FINAL)")
print("="*80)

models = ['cv_normal', 'cv_t', 'garch_normal', 'garch_t']
model_names = {
    'cv_normal': 'CV Normal',
    'cv_t': 'CV Student-t',
    'garch_normal': 'GARCH Normal',
    'garch_t': 'GARCH Student-t'
}

results_table = []

for model_key in models:
    print(f"\nProcessing {model_names[model_key]}...")
    
    # Christoffersen test
    clr = christoffersen_test(actual_returns, forecasts[model_key]['var'], alpha=0.01)
    print(f"  Violations: {clr['n_violations']}")
    print(f"  Transitions: n01={clr['transitions']['n01']}, n11={clr['transitions']['n11']}")
    print(f"  CLR_Ind: LR={clr['LR_ind']:.4f}, p={clr['p_value_ind']:.4f}")
    
    # Berkowitz test
    if model_key in ['cv_t', 'garch_t']:
        nu_avg = forecasts[model_key]['nu'].mean()
        blr = berkowitz_test(actual_returns, forecasts[model_key]['mu'],
                            forecasts[model_key]['sigma'], distribution='t', df=nu_avg)
    else:
        blr = berkowitz_test(actual_returns, forecasts[model_key]['mu'],
                            forecasts[model_key]['sigma'], distribution='norm')
    
    print(f"  BLR: z_mean={blr['z_mean']:.4f}, z_var={blr['z_var']:.4f}")
    print(f"  BLR_tail: LR={blr['LR_tail']:.4f}, p={blr['p_value_tail']:.4f}")
    
    results_table.append({
        'Model': model_names[model_key],
        'Violation': clr['n_violations'],
        'CLR_UC': clr['p_value_uc'],
        'CLR_Ind': clr['p_value_ind'],
        'CLR_Joint': clr['p_value_cc'],
        'BLR_tail': blr['p_value_tail'],
        'BLR_Ind': blr['p_value_ind']
    })

df_results = pd.DataFrame(results_table)

print("\n" + "="*80)
print("FINAL RESULTS TABLE")
print("="*80)
print(df_results.to_string(index=False))

print("\n" + "="*80)
print("INTERPRETATION (α = 0.01)")
print("="*80)
print("REJECT if p-value < 0.01\n")
for _, row in df_results.iterrows():
    print(f"{row['Model']:20} | CLR: {'REJECT' if row['CLR_Joint'] < 0.01 else 'ACCEPT':6} | "
          f"BLR: {'REJECT' if row['BLR_tail'] < 0.01 else 'ACCEPT':6}")
    
# ============================================
# AVERAGE RELATIVE DIFFERENCE (ARD)
# ============================================

eps = 1e-12  # guard against divide-by-zero

# Convert to loss magnitudes (positive numbers)
LVaR = {
    'cv_n':  -forecasts['cv_normal']['var'],
    'cv_t':  -forecasts['cv_t']['var'],
    'ga_n':  -forecasts['garch_normal']['var'],
    'ga_t':  -forecasts['garch_t']['var'],
}
LES = {
    'cv_n':  -forecasts['cv_normal']['es'],
    'cv_t':  -forecasts['cv_t']['es'],
    'ga_n':  -forecasts['garch_normal']['es'],
    'ga_t':  -forecasts['garch_t']['es'],
}

# Per-day ARD series
ARD_VaR_CV     = (LVaR['cv_t'] - LVaR['cv_n']) / np.maximum(LVaR['cv_n'], eps)
ARD_VaR_GARCH  = (LVaR['ga_t'] - LVaR['ga_n']) / np.maximum(LVaR['ga_n'], eps)

ARD_ES_CV      = (LES['cv_t'] - LES['cv_n']) / np.maximum(LES['cv_n'], eps)
ARD_ES_GARCH   = (LES['ga_t'] - LES['ga_n']) / np.maximum(LES['ga_n'], eps)

# Time-averaged ARD (the “Expectation”)
ARD_summary = pd.DataFrame({
    'ARD_VaR_CV_mean'    : [ARD_VaR_CV.mean()],
    'ARD_VaR_CV_std'     : [ARD_VaR_CV.std(ddof=1)],
    'ARD_VaR_GARCH_mean' : [ARD_VaR_GARCH.mean()],
    'ARD_VaR_GARCH_std'  : [ARD_VaR_GARCH.std(ddof=1)],
    'ARD_ES_CV_mean'     : [ARD_ES_CV.mean()],
    'ARD_ES_CV_std'      : [ARD_ES_CV.std(ddof=1)],
    'ARD_ES_GARCH_mean'  : [ARD_ES_GARCH.mean()],
    'ARD_ES_GARCH_std'   : [ARD_ES_GARCH.std(ddof=1)],
})

print("\nAverage Relative Difference (ARD) — time-averaged over rolling horizon")
print(ARD_summary.to_string(index=False))
