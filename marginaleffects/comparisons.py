# TODO: Sanitize data: pandas, polars, or numpy array
from .by import *
from .utils import *
from .sanity import *
from .hypothesis import *
from .uncertainty import *
from .sanitize_variables import *
import polars as pl
import pandas as pd
import numpy as np
import patsy
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm


estimands = {
    "difference": lambda hi, lo, eps, x, y: hi - lo,
    "differenceavg": lambda hi, lo, eps, x, y: np.array([np.mean(hi - lo)]),
    # "differenceavgwts": lambda hi, lo, w: (hi * w).sum() / w.sum() - (lo * w).sum() / w.sum(),

    "dydx": lambda hi, lo, eps, x, y: (hi - lo) / eps,
    "eyex": lambda hi, lo, eps, x, y: (hi - lo) / eps * (x / y),
    "eydx": lambda hi, lo, eps, x, y: ((hi - lo) / eps) / y,
    "dyex": lambda hi, lo, eps, x, y: ((hi - lo) / eps) * x,

    "dydxavg": lambda hi, lo, eps, x, y: ((hi - lo) / eps).mean(),
    "eyexavg": lambda hi, lo, eps, x, y: ((hi - lo) / eps * (x / y)).mean(),
    "eydxavg": lambda hi, lo, eps, x, y: (((hi - lo) / eps) / y).mean(),
    "dyexavg": lambda hi, lo, eps, x, y: (((hi - lo) / eps) * x).mean(),
    # "dydxavgwts": lambda hi, lo, eps, w: (((hi - lo) / eps) * w).sum() / w.sum(),
    # "eyexavgwts": lambda hi, lo, eps, y, x, w: (((hi - lo) / eps) * (x / y) * w).sum() / w.sum(),
    # "eydxavgwts": lambda hi, lo, eps, y, x, w: ((((hi - lo) / eps) / y) * w).sum() / w.sum(),
    # "dyexavgwts": lambda hi, lo, eps, x, w: (((hi - lo) / eps) * x * w).sum() / w.sum(),

    "ratio": lambda hi, lo, eps, x, y: hi / lo,
    "ratioavg": lambda hi, lo, eps, x, y: np.array([np.mean(hi) / np.mean(lo)]),
    # "ratioavgwts": lambda hi, lo, w: (hi * w).sum() / w.sum() / (lo * w).sum() / w.sum(),

    "lnratio": lambda hi, lo, eps, x, y: np.log(hi / lo),
    "lnratioavg": lambda hi, lo, eps, x, y: np.array([np.log(np.mean(hi) / np.mean(lo))]),
    # "lnratioavgwts": lambda hi, lo, w: np.log((hi * w).sum() / w.sum() / (lo * w).sum() / w.sum()),

    "lnor": lambda hi, lo, eps, x, y: np.log((hi / (1 - hi)) / (lo / (1 - lo))),
    "lnoravg": lambda hi, lo, eps, x, y: np.log((np.mean(hi) / (1 - np.mean(hi))) / (np.mean(lo) / (1 - np.mean(lo)))),
    # "lnoravgwts": lambda hi, lo, w: np.log(((hi * w).sum() / w.sum() / (1 - (hi * w).sum() / w.sum())) / ((lo * w).sum() / w.sum() / (1 - (lo * w).sum() / w.sum()))),

    "lift": lambda hi, lo, eps, x, y: (hi - lo) / lo,
    "liftavg": lambda hi, lo, eps, x, y: np.array([(np.mean(hi) - np.mean(lo)) / np.mean(lo)]),

    "expdydx": lambda hi, lo, eps, x, y: ((np.exp(hi) - np.exp(lo)) / np.exp(eps)) / eps,
    "expdydxavg": lambda hi, lo, eps, x, y: (((np.exp(hi) - np.exp(lo)) / np.exp(eps)) / eps).mean(),
    # "expdydxavgwts": lambda hi, lo, eps, w: ((((np.exp(hi) - np.exp(lo)) / np.exp(eps)) / eps) * w).sum() / w.sum(),
}


def get_exog(model, variable, newdata):
    lo = newdata.clone().with_columns(variable.lo.alias(variable.variable))
    hi = newdata.clone().with_columns(variable.hi.alias(variable.variable))
    # pad for character predictors
    if variable.pad is not None:
        pad = pl.concat([newdata.slice(0, 1)] * variable.pad.len())
        pad = pad.with_columns(variable.pad.alias(variable.variable))
        lo = pl.concat([lo, pad])
        hi = pl.concat([hi, pad])
    y, lo = patsy.dmatrices(model.model.formula, lo.to_pandas())
    y, hi = patsy.dmatrices(model.model.formula, hi.to_pandas())
    # unpad
    if variable.pad is not None:
        lo = lo[pad.shape[0]:]
        hi = hi[pad.shape[0]:]
    return hi, lo


def get_estimand(model, params, hi, lo, comparison, variabletype = "numeric", eps = None, df = None, by = None, x = None, y = None):
    p_hi = model.model.predict(params, hi)
    p_lo = model.model.predict(params, lo)
    if variabletype != "numeric" and comparison in ["dydx", "eyex", "eydx", "dyex", "dydxavg", "eyexavg", "eydxavg", "dyexavg"]:
        fun = estimands["difference"]
    else:
        fun = estimands[comparison]
    out = fun(hi = p_hi, lo = p_lo, eps = eps, x = x, y = y)
    return out


def get_comparison(
        model,
        variable,
        newdata,
        comparison,
        vcov,
        conf_int,
        by,
        hypothesis,
        eps):

    # predictors
    hi, lo = get_exog(model, variable=variable, newdata=newdata)

    variabletype = get_one_variable_type(variable = variable.variable, newdata = newdata)

    xvar = newdata[variable.variable]
    yvar = newdata[model.model.endog_names]

    # estimands
    def fun(x):
        out = get_estimand(model, x, hi, lo, comparison=comparison, variabletype = variabletype, eps = eps, x = xvar, y = yvar)
        out = get_by(model, out, newdata=newdata, by=by)
        out = get_hypothesis(out, hypothesis=hypothesis)
        return out
    out = fun(np.array(model.params))

    # uncetainty
    if vcov is not None:
        J = get_jacobian(fun, model.params.to_numpy())
        se = get_se(J, vcov)
        out = out.with_columns(pl.Series(se).alias("std_error"))

    # output
    out = out.with_columns(pl.Series([variable.variable]).alias("term"))
    out = out.with_columns(pl.Series([variable.lab]).alias("contrast"))
    return out


def comparisons(
        model,
        variables = None,
        newdata = None,
        comparison = "differenceavg",
        vcov = True,
        conf_int = 0.95,
        by = None,
        hypothesis = None,
        eps = 1e-4):
    """
    Comparisons Between Predictions Made With Different Regressor Values

    Predict the outcome variable at different regressor values (e.g., college graduates vs. others), and compare those predictions by computing a difference, ratio, or some other function. `comparisons()` can return many quantities of interest, such as contrasts, differences, risk ratios, changes in log odds, lift, 

    Parameters
    ----------
    * model : `statsmodels.formula.api` modelted model
    * conf_int : float
    * vcov : bool or string which corresponds to one of the attributes in `model`. "HC3" looks for an attributed named `cov_HC3`.
    * newdata : None, DataFrame or `datagrid()` call.
    * hypothesis : Numpy array for linear combinations. 
    * comparison : "difference", "differenceavg", "ratio", "ratioavg", "lnratio", "lnratioavg", "lnor", "lnoravg", "lift", "liftavg", "expdydx", "expdydxavg", "expdydxavgwts"
    * by : None, string, or list of strings
    """


    # sanity
    V = sanitize_vcov(vcov, model)
    newdata = sanitize_newdata(model, newdata)

    # after sanitize_newdata() 
    variables = sanitize_variables(variables=variables, model=model, newdata=newdata, comparison=comparison, eps=eps)

    # computation
    out = []
    for v in variables:
        tmp = get_comparison(
            model,
            variable=v,
            newdata=newdata,
            comparison=comparison,
            vcov=V,
            conf_int=conf_int,
            by=by,
            hypothesis=hypothesis,
            eps=eps)
        out.append(tmp)
    out = pl.concat(out)

    # uncertainty
    out = get_z_p_ci(out, model, conf_int=conf_int)

    # output
    out = sort_columns(out, by = by)
    return out