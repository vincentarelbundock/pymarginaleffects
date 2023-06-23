# TODO: Sanitize data: pandas, polars, or numpy array
from .by import *
from warnings import warn
from .utils import *
from .hypothesis import *
from .uncertainty import *
import polars as pl
import pandas as pd
import numpy as np
import patsy
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm


estimands = dict(
    difference=lambda hi, lo: hi - lo,
    differenceavg=lambda hi, lo: np.array([np.mean(hi - lo)]),
    ratio=lambda hi, lo: hi / lo,
    ratioavg=lambda hi, lo: np.array([np.mean(hi / lo)])
)


def get_comparison_exog_numeric(fit, variable, value, data):
    lo = data.clone().with_columns(pl.col(variable) - value/2)
    hi = data.clone().with_columns(pl.col(variable) + value/2)
    y, lo = patsy.dmatrices(fit.model.formula, lo)
    y, hi = patsy.dmatrices(fit.model.formula, hi)
    return hi, lo


def get_estimand(fit, params, hi, lo, comparison, df = None, by = None):
    p_hi = fit.model.predict(params, hi)
    p_lo = fit.model.predict(params, lo)
    fun = estimands[comparison]
    out = fun(p_hi, p_lo)
    return out


def get_variables(variables, fit, newdata):
    if variables is None:
        variables = fit.model.exog_names
        variables = [x for x in variables if x in newdata.columns]
        variables = dict(zip(variables, [1] * len(variables)))
    elif isinstance(variables, str):
        variables = {variables: 1}
    elif isinstance(variables, list):
        variables = dict(zip(variables, [1] * len(variables)))
    else:
        assert isinstance(variables, dict), "`variables` must be None, a dict, string, or list of strings"

    for key in variables.keys():
        if key not in newdata.columns:
            del variables[key]
            warn(f"Variable `{key}` not in newdata")

    if not variables:
        raise ValueError("There is no valid column name in `variables`.")
    return variables


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
    hi, lo = get_comparison_exog_numeric(fit, variable=variable, value=value, data=newdata)

    # estimands
    def fun(x):
        out = get_estimand(fit, x, hi, lo, comparison=comparison)
        out = get_by(fit, out, df=newdata, by=by)
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
    out = out.with_columns(pl.Series([variable]).alias("term"))
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

    if newdata is None:
        newdata = pl.from_pandas(fit.model.data.frame)

    # after newdata sanitation
    variables = get_variables(variables=variables, fit=fit, newdata=newdata)

    # computation
    out = []
    for v in variables:
        tmp = get_comparison(
            fit,
            variable=v,
            value=variables.get(v),
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
