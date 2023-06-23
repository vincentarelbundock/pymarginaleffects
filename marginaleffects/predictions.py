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


def get_estimand(fit, params, hi, lo, comparison, df = None, by = None):
    p_hi = fit.model.predict(params, hi)
    p_lo = fit.model.predict(params, lo)
    fun = estimands[comparison]
    out = fun(p_hi, p_lo)
    return out

def get_exog(fit, newdata = None):
    if newdata is None:
        newdata = fit.model.data.frame
    y, out = patsy.dmatrices(fit.model.formula, newdata)
    return out

def predictions(fit, conf_int = 0.95, by = None, newdata = None):
    # predictors
    if newdata is None:
        newdata = pl.from_pandas(fit.model.data.frame)
    exog = get_exog(fit, newdata = newdata)
    # estimands
    def fun(x):
        out = fit.model.predict(x, exog)
        out = get_by(fit, out, df=newdata, by=by)
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
