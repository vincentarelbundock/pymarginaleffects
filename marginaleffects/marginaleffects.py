import pandas as pd
import numpy as np
import patsy
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm


def get_comparison_exog_numeric(fit, variable, value, data):
    lo = data.copy()
    hi = data.copy()
    lo[variable] = lo[variable] - value/2
    hi[variable] = hi[variable] + value/2
    y, lo = patsy.dmatrices(fit.model.formula, lo)
    y, hi = patsy.dmatrices(fit.model.formula, hi)
    return hi, lo

estimands = dict(
    difference=lambda hi, lo: hi - lo,
    differenceavg=lambda hi, lo: np.array([np.mean(hi - lo)]),
    ratio=lambda hi, lo: hi / lo,
    ratioavg=lambda hi, lo: np.array([np.mean(hi / lo)])
)

def get_estimand(fit, params, hi, lo, comparison):
    p_hi = fit.model.predict(params, hi)
    p_lo = fit.model.predict(params, lo)
    fun = estimands[comparison]
    out = fun(p_hi, p_lo)
    return out

def get_jacobian(func, x):
    # forward finite difference (faster)
    eps = max(1e-8, 1e-4 * np.min(np.abs(x)))
    baseline = func(x)
    df = np.empty((len(baseline), len(x)), dtype=np.float64)
    for i, xi in enumerate(x):
        dx = x.copy()
        dx[i] = dx[i] + eps
        df[:, i] = (func(dx) - baseline) / eps
    return df

def get_se(J, V):
    se = np.sqrt(np.sum((J @ V) * J, axis=1))
    return se

def get_z_p_ci(df, fit, conf_int):
    df['statistic'] = df['estimate'] / df['std_error']
    if hasattr(fit, 'df_resid') and isinstance(fit.df_resid, float):
        dof = fit.df_resid
    else:
        dof = np.Inf
    critical_value = stats.t.ppf((1 + conf_int) / 2, dof)
    df['conf_low'] = df['estimate'] - critical_value * df['std_error']
    df['conf_high'] = df['estimate'] + critical_value * df['std_error']
    df['p_value'] = 2 * (1 - stats.t.cdf(np.abs(df['statistic']), dof))
    return df

def comparisons(fit, variable, value = 1, comparison = "difference", conf_int = 0.95):
    # predictors
    df = fit.model.data.frame
    hi, lo = get_comparison_exog_numeric(fit, variable=variable, value=value, data=df)
    # estimands
    def fun(x):
        return get_estimand(fit, x, hi, lo, comparison=comparison)
    b = fun(np.array(fit.params))
    # uncertainty
    J = get_jacobian(fun, fit.params.to_numpy())
    V = fit.cov_params()
    se = get_se(J, V)
    out = pd.DataFrame(data={"estimate": b, "std_error": se})
    out = get_z_p_ci(out, fit, conf_int=conf_int)
    return out
