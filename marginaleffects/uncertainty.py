import pandas as pd
import numpy as np
import polars as pl
import patsy
import scipy.stats as stats

def get_jacobian(func, x):
    # forward finite difference (faster)
    eps = max(1e-8, 1e-4 * np.min(np.abs(x)))
    baseline = func(x)["estimate"]
    out = np.empty((len(baseline), len(x)), dtype=np.float64)
    for i, xi in enumerate(x):
        dx = x.copy()
        dx[i] = dx[i] + eps
        out[:, i] = (func(dx)["estimate"] - baseline) / eps
    return out

def get_se(J, V):
    se = np.sqrt(np.sum((J @ V) * J, axis=1))
    return se

def get_z_p_ci(df, fit, conf_int):
    df = df.with_columns((pl.col("estimate") / pl.col("std_error")).alias("statistic"))
    if hasattr(fit, 'df_resid') and isinstance(fit.df_resid, float):
        dof = fit.df_resid
    else:
        dof = np.Inf
    critical_value = stats.t.ppf((1 + conf_int) / 2, dof)

    df = df.with_columns((pl.col("estimate") - critical_value * pl.col("std_error")).alias("conf_low"))
    df = df.with_columns((pl.col("estimate") + critical_value * pl.col("std_error")).alias("conf_high"))
    # df = df.with_columns((2 * (1 - stats.t.cdf(pl.col("statistic").abs()), dof)).alias("p_value"))
    return df

