import warnings

import numpy as np
import polars as pl
import scipy.stats as stats


def get_jacobian(func, coefs, eps_vcov=None):
    # forward finite difference (faster)
    if coefs.ndim == 2:
        if isinstance(coefs, np.ndarray):
            coefs_flat = coefs.flatten()
        else:
            coefs_flat = coefs.to_numpy().flatten()
        baseline = func(coefs)["estimate"].to_numpy()
        jac = np.empty((baseline.shape[0], len(coefs_flat)), dtype=np.float64)
        for i, xi in enumerate(coefs_flat):
            if eps_vcov is not None:
                h = eps_vcov
            else:
                h = max(abs(xi) * np.sqrt(np.finfo(float).eps), 1e-10)
            dx = np.copy(coefs_flat)
            dx[i] = dx[i] + h
            tmp = dx.reshape(coefs.shape)
            jac[:, i] = (func(tmp)["estimate"].to_numpy() - baseline) / h
        return jac
    else:
        baseline = func(coefs)["estimate"].to_numpy()
        jac = np.empty((baseline.shape[0], len(coefs)), dtype=np.float64)
        for i, xi in enumerate(coefs):
            if eps_vcov is not None:
                h = eps_vcov
            else:
                h = max(abs(xi) * np.sqrt(np.finfo(float).eps), 1e-10)
            dx = np.copy(coefs)
            dx[i] = dx[i] + h
            jac[:, i] = (func(dx)["estimate"].to_numpy() - baseline) / h
        return jac


def get_se(J, V):
    # J are different in python versus R
    '''Python
    array([[ 3.26007929e-03, -1.11398030e+05, -1.07005746e+07,
         7.82170746e+07,  2.78495302e+07,  1.34757804e+00],
       [ 3.00860803e+05,  1.11398009e+05,  3.15899457e+07,
         4.00666771e+07,  1.42659178e+07, -5.02687187e+00],
       [-3.00860806e+05,  2.03275902e-02, -2.08893711e+07,
        -1.18283752e+08, -4.21154480e+07,  3.67929389e+00],
       ...,
       [ 1.94566192e-02,  6.31280874e+04,  6.06390551e+06,
         6.95824154e+07,  2.47751212e+07,  4.27175339e-01],
       [ 1.19234156e+05, -6.31280929e+04,  1.25194134e+07,
        -2.27053711e+07, -8.08434379e+06, -1.51166669e+00],
       [-1.19234176e+05,  5.47722901e-03, -1.85833189e+07,
        -4.68770444e+07, -1.66907774e+07,  1.08449133e+00]])
    '''
    '''R
                 [,1]          [,2]          [,3]          [,4]          [,5]          [,6]
   [1,]  4.211085e-03  0.1129098874  7.622065e-01  6.449953e-03  2.390449e-01  1.167441e+00
   [2,]  2.317994e-03 -0.0002889116  4.311471e-01  1.240511e-02  4.599770e-01  2.307350e+00
   [3,] -1.458913e-02 -0.7474941310 -2.844880e+00  2.018562e-02  7.334876e-01  3.936196e+00
   [4,] -1.768363e-02 -0.7516309380 -3.412940e+00  1.977302e-02  5.853414e-01  3.816193e+00
   [5,] -6.148701e-03 -0.3694016210 -1.168253e+00  1.883557e-02  6.798748e-01  3.578758e+00
   [6,]  3.851085e-03  0.0970890213  6.970463e-01  6.842971e-03  2.519078e-01  1.238578e+00
   [7,] -1.807982e-02 -0.8520217380 -3.525564e+00  2.090703e-02  7.173019e-01  4.076872e+00
   [8,] -1.324527e-02 -0.5110488046 -2.556336e+00  1.326720e-02  2.654043e-01  2.560569e+00
   [9,]  7.328301e-03  0.1888463791  1.392377e+00  1.094683e-02  4.361391e-01  2.079898e+00
  [10,] -4.511196e-03 -0.2639279320 -8.390823e-01  1.642779e-02  5.682850e-01  3.055570e+00
  ...
    '''
    se = np.sqrt(np.sum((J @ V) * J, axis=1))
    return se


def get_z_p_ci(df, model, conf_level, hypothesis_null=0):
    if "std_error" not in df.columns:
        return df
    df = df.with_columns(
        ((pl.col("estimate") - float(hypothesis_null)) / pl.col("std_error")).alias(
            "statistic"
        )
    )
    if hasattr(model, "df_resid") and isinstance(model.df_resid, float):
        dof = model.df_resid
    else:
        dof = np.inf
    critical_value = stats.t.ppf((1 + conf_level) / 2, dof)

    df = df.with_columns(
        (pl.col("estimate") - critical_value * pl.col("std_error")).alias("conf_low")
    )
    df = df.with_columns(
        (pl.col("estimate") + critical_value * pl.col("std_error")).alias("conf_high")
    )

    df = df.with_columns(
        pl.col("statistic")
        .map_batches(
            lambda x: (2 * (1 - stats.t.cdf(np.abs(x), dof))), return_dtype=pl.Float64
        )
        .alias("p_value")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            df = df.with_columns(
                pl.col("p_value")
                .map_batches(lambda x: -np.log2(x), return_dtype=pl.Float64)
                .alias("s_value")
            )
        except Exception as e:
            print(f"An exception occurred: {e}")
    return df
