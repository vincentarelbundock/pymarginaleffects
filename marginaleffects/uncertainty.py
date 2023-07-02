import warnings

import numpy as np
import patsy
import polars as pl
import scipy.stats as stats


def get_jacobian(func, coefs):
    if coefs.ndim == 2:
        if isinstance(coefs, np.ndarray):
            coefs_flat = coefs.flatten()
        else:
            coefs_flat = coefs.to_numpy().flatten()

        eps = max(1e-8, 1e-4 * np.min(np.abs(coefs_flat)))
        baseline = func(coefs)["estimate"].to_numpy()
        jac = np.empty((baseline.shape[0], len(coefs_flat)), dtype=np.float64)
        for i, xi in enumerate(coefs_flat):
            dx = np.copy(coefs_flat)
            dx[i] = dx[i] + eps
            tmp = dx.reshape(coefs.shape)
            jac[:, i] = (func(tmp)["estimate"].to_numpy() - baseline) / eps
        return jac
    else:
        # forward finite difference (faster)
        eps = max(1e-8, 1e-4 * np.min(np.abs(coefs)))
        baseline = func(coefs)["estimate"].to_numpy()
        jac = np.empty((baseline.shape[0], len(coefs)), dtype=np.float64)
        for i, xi in enumerate(coefs):
            dx = np.copy(coefs)
            dx[i] = dx[i] + eps
            jac[:, i] = (func(dx)["estimate"].to_numpy() - baseline) / eps
        return jac


def get_se(J, V):
    se = np.sqrt(np.sum((J @ V) * J, axis=1))
    return se


def get_z_p_ci(df, model, conf_int):
    if "std_error" not in df.columns:
        return df
    df = df.with_columns((pl.col("estimate") / pl.col("std_error")).alias("statistic"))
    if hasattr(model, "df_resid") and isinstance(model.df_resid, float):
        dof = model.df_resid
    else:
        dof = np.Inf
    critical_value = stats.t.ppf((1 + conf_int) / 2, dof)

    df = df.with_columns(
        (pl.col("estimate") - critical_value * pl.col("std_error")).alias("conf_low")
    )
    df = df.with_columns(
        (pl.col("estimate") + critical_value * pl.col("std_error")).alias("conf_high")
    )
    df = df.with_columns(
        pl.col("statistic")
        .apply(lambda x: (2 * (1 - stats.t.cdf(np.abs(x), dof))))
        .alias("p_value")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            df = df.with_columns(
                pl.col("p_value").apply(lambda x: -np.log2(x)).alias("s_value")
            )
        except:
            pass
    return df
