# TODO: Sanitize data: pandas, polars, or numpy array

from .uncertainty import *
from .by import *
from .utils import *
import polars as pl
import pandas as pd
import numpy as np
import patsy
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm


def get_comparison_exog_numeric(fit, variable, value, data):
    lo = data.clone().with_columns(pl.col(variable) - value/2)
    hi = data.clone().with_columns(pl.col(variable) + value/2)
    y, lo = patsy.dmatrices(fit.model.formula, lo)
    y, hi = patsy.dmatrices(fit.model.formula, hi)
    return hi, lo


estimands = dict(
    difference=lambda hi, lo: hi - lo,
    differenceavg=lambda hi, lo: np.array([np.mean(hi - lo)]),
    ratio=lambda hi, lo: hi / lo,
    ratioavg=lambda hi, lo: np.array([np.mean(hi / lo)])
)


def get_estimand(fit, params, hi, lo, comparison, df = None, by = None):
    p_hi = fit.model.predict(params, hi)
    p_lo = fit.model.predict(params, lo)
    fun = estimands[comparison]
    out = fun(p_hi, p_lo)
    return out


def comparisons(fit, variable, value = 1, comparison = "difference", conf_int = 0.95, by = None):
    # predictors
    df = pl.from_pandas(fit.model.data.frame)
    hi, lo = get_comparison_exog_numeric(fit, variable=variable, value=value, data=df)
    # estimands
    def fun(x):
        out = get_estimand(fit, x, hi, lo, comparison=comparison)
        out = get_by(fit, out, df=df, by=by)
        return out
    out = fun(np.array(fit.params))
    # uncertainty
    J = get_jacobian(fun, fit.params.to_numpy())
    V = fit.cov_params()
    se = get_se(J, V)
    out = out.with_columns(pl.Series(se).alias("std_error"))
    out = get_z_p_ci(out, fit, conf_int=conf_int)
    out = sort_columns(out, by = by)
    return out
