from .uncertainty import *
from .sanity import *
from .by import *
from .utils import *
from .hypothesis import *
import polars as pl
import pandas as pd
import numpy as np
import patsy
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

def get_exog(fit, newdata = None):
    y, out = patsy.dmatrices(fit.model.formula, newdata.to_pandas())
    return out

def predictions(fit, conf_int = 0.95, vcov = True, by = None, newdata = None, hypothesis = None):
    """
    Predictions

    Outcome predicted by a fitted model on a specified scale for a given combination of values of the predictor variables,
    such as their observed values, their means, or factor levels (a.k.a. "reference grid").

    Parameters
    ----------
    model : `statsmodels.formula.api` fitted model
    conf_int : float
    vcov : bool or string which corresponds to one of the attributes in `fit`. "HC3" looks for an attributed named `cov_HC3`.
    newdata : None, DataFrame or `datagrid()` call.
    hypothesis : Numpy array for linear combinations. 
    """

    # sanity checks
    V = sanitize_vcov(vcov, fit)
    newdata = sanitize_newdata(fit, newdata)

    # predictors
    exog = get_exog(fit, newdata = newdata)

    # estimands
    def fun(x):
        out = fit.model.predict(x, exog)
        out = get_by(fit, out, newdata=newdata, by=by)
        out = get_hypothesis(out, hypothesis=hypothesis)
        return out
    out = fun(np.array(fit.params))
    if vcov is not None:
        J = get_jacobian(fun, fit.params.to_numpy())
        se = get_se(J, V)
        out = out.with_columns(pl.Series(se).alias("std_error"))
        out = get_z_p_ci(out, fit, conf_int=conf_int)
    out = sort_columns(out, by = by)
    return out
