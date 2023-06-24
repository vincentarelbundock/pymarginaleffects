# TODO: Sanitize data: pandas, polars, or numpy array
from .by import *
from warnings import warn
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
    "difference": lambda hi, lo: hi - lo,
    "differenceavg": lambda hi, lo: np.array([np.mean(hi - lo)]),
    # "differenceavgwts": lambda hi, lo, w: (hi * w).sum() / w.sum() - (lo * w).sum() / w.sum(),

    # "dydx": lambda hi, lo, eps: (hi - lo) / eps,
    # "eyex": lambda hi, lo, eps, y, x: (hi - lo) / eps * (x / y),
    # "eydx": lambda hi, lo, eps, y, x: ((hi - lo) / eps) / y,
    # "dyex": lambda hi, lo, eps, x: ((hi - lo) / eps) * x,

    # "dydxavg": lambda hi, lo, eps: ((hi - lo) / eps).mean(),
    # "eyexavg": lambda hi, lo, eps, y, x: ((hi - lo) / eps * (x / y)).mean(),
    # "eydxavg": lambda hi, lo, eps, y, x: (((hi - lo) / eps) / y).mean(),
    # "dyexavg": lambda hi, lo, eps, x: (((hi - lo) / eps) * x).mean(),
    # "dydxavgwts": lambda hi, lo, eps, w: (((hi - lo) / eps) * w).sum() / w.sum(),
    # "eyexavgwts": lambda hi, lo, eps, y, x, w: (((hi - lo) / eps) * (x / y) * w).sum() / w.sum(),
    # "eydxavgwts": lambda hi, lo, eps, y, x, w: ((((hi - lo) / eps) / y) * w).sum() / w.sum(),
    # "dyexavgwts": lambda hi, lo, eps, x, w: (((hi - lo) / eps) * x * w).sum() / w.sum(),

    "ratio": lambda hi, lo: hi / lo,
    "ratioavg": lambda hi, lo: np.array([np.mean(hi) / np.mean(lo)]),
    # "ratioavgwts": lambda hi, lo, w: (hi * w).sum() / w.sum() / (lo * w).sum() / w.sum(),

    "lnratio": lambda hi, lo: np.log(hi / lo),
    "lnratioavg": lambda hi, lo: np.array([np.log(np.mean(hi) / np.mean(lo))]),
    # "lnratioavgwts": lambda hi, lo, w: np.log((hi * w).sum() / w.sum() / (lo * w).sum() / w.sum()),

    "lnor": lambda hi, lo: np.log((hi / (1 - hi)) / (lo / (1 - lo))),
    "lnoravg": lambda hi, lo: np.log((np.mean(hi) / (1 - np.mean(hi))) / (np.mean(lo) / (1 - np.mean(lo)))),
    # "lnoravgwts": lambda hi, lo, w: np.log(((hi * w).sum() / w.sum() / (1 - (hi * w).sum() / w.sum())) / ((lo * w).sum() / w.sum() / (1 - (lo * w).sum() / w.sum()))),

    "lift": lambda hi, lo: (hi - lo) / lo,
    "liftavg": lambda hi, lo: np.array([(np.mean(hi) - np.mean(lo)) / np.mean(lo)]),

    # "expdydx": lambda hi, lo, eps: ((np.exp(hi) - np.exp(lo)) / np.exp(eps)) / eps,
    # "expdydxavg": lambda hi, lo, eps: (((np.exp(hi) - np.exp(lo)) / np.exp(eps)) / eps).mean(),
    # "expdydxavgwts": lambda hi, lo, eps, w: ((((np.exp(hi) - np.exp(lo)) / np.exp(eps)) / eps) * w).sum() / w.sum(),
}


def get_exog(fit, variable, newdata):
    lo = newdata.clone().with_columns(variable.lo.alias(variable.variable))
    hi = newdata.clone().with_columns(variable.hi.alias(variable.variable))
    # pad for character predictors
    if variable.pad is not None:
        pad = pl.concat([newdata.slice(0, 1)] * variable.pad.len())
        pad = pad.with_columns(variable.pad.alias(variable.variable))
        lo = pl.concat([lo, pad])
        hi = pl.concat([hi, pad])
    y, lo = patsy.dmatrices(fit.model.formula, lo)
    y, hi = patsy.dmatrices(fit.model.formula, hi)
    # unpad
    if variable.pad is not None:
        lo = lo[pad.shape[0]:]
        hi = hi[pad.shape[0]:]
    return hi, lo


def get_estimand(fit, params, hi, lo, comparison, df = None, by = None):
    p_hi = fit.model.predict(params, hi)
    p_lo = fit.model.predict(params, lo)
    fun = estimands[comparison]
    out = fun(p_hi, p_lo)
    return out


def get_comparison(
        fit,
        variable,
        newdata,
        value = 1,
        comparison = "difference",
        vcov = True,
        conf_int = 0.95,
        by = None,
        hypothesis = None):

    # predictors
    hi, lo = get_exog(fit, variable=variable, newdata=newdata)

    # estimands
    def fun(x):
        out = get_estimand(fit, x, hi, lo, comparison=comparison)
        out = get_by(fit, out, newdata=newdata, by=by)
        out = get_hypothesis(out, hypothesis=hypothesis)
        return out
    out = fun(np.array(fit.params))

    # uncetainty
    if vcov:
        J = get_jacobian(fun, fit.params.to_numpy())
        V = fit.cov_params()
        se = get_se(J, V)
        out = out.with_columns(pl.Series(se).alias("std_error"))

    # output
    out = out.with_columns(pl.Series([variable.variable]).alias("term"))
    out = out.with_columns(pl.Series([variable.lab]).alias("contrast"))
    return out


def comparisons(
        fit,
        variables = None,
        newdata = None,
        value = 1,
        comparison = "differenceavg",
        vcov = True,
        conf_int = 0.95,
        by = None,
        hypothesis = None):

    # sanity
    assert isinstance(vcov, bool), "`vcov` must be a boolean"

    if vcov is True and (by is not None or hypothesis is not None):
        vcov = False
        warn("vcov is set to False because `by` or `hypothesis` is not None")

    newdata = sanitize_newdata(fit, newdata)

    # after newdata sanitation
    variables = sanitize_variables(variables=variables, fit=fit, newdata=newdata)

    # computation
    out = []
    for v in variables:
        tmp = get_comparison(
            fit,
            variable=v,
            newdata=newdata,
            comparison=comparison,
            vcov=vcov,
            conf_int=conf_int,
            by=by,
            hypothesis=hypothesis)
        out.append(tmp)
    out = pl.concat(out)

    # uncertainty
    out = get_z_p_ci(out, fit, conf_int=conf_int)

    # output
    out = sort_columns(out, by = by)
    return out