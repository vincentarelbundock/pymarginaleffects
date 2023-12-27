import re
from functools import reduce

import numpy as np
import patsy
import polars as pl

from .classes import MarginaleffectsDataFrame
from .equivalence import get_equivalence
from .estimands import estimands
from .hypothesis import get_hypothesis
from .sanitize_model import sanitize_model
from .sanity import (
    sanitize_by,
    sanitize_hypothesis_null,
    sanitize_newdata,
    sanitize_variables,
    sanitize_vcov,
)
from .transform import get_transform
from .uncertainty import get_jacobian, get_se, get_z_p_ci
from .utils import get_pad, sort_columns, upcast
from .model_pyfixest import ModelPyfixest


def comparisons(
    model,
    variables=None,
    newdata=None,
    comparison="difference",
    vcov=True,
    conf_level=0.95,
    by=False,
    wts=None,
    hypothesis=None,
    equivalence=None,
    transform=None,
    eps=1e-4,
    eps_vcov=None,
):
    """
    `comparisons()` and `avg_comparisons()` are functions for predicting the outcome variable at different regressor values and comparing those predictions by computing a difference, ratio, or some other function. These functions can return many quantities of interest, such as contrasts, differences, risk ratios, changes in log odds, lift, slopes, elasticities, etc.

    # Usage:

        comparisons(model, variables = NULL, newdata = NULL, comparison = "difference",
                    transform = NULL, equivalence = NULL, by = FALSE, cross = FALSE,
                    type = "response", hypothesis = 0, conf.level = 0.95, ...)

        avg_comparisons(model, variables = NULL, newdata = NULL, comparison = "difference",
                        transform = NULL, equivalence = NULL, by = FALSE, cross = FALSE,
                        type = "response", hypothesis = 0, conf.level = 0.95, ...)

    # Args:

    - model (object): a model object fitted using the `statsmodels` formula API.
    - variables (str, list, or dictionary): a string, list of strings, or dictionary of variables to compute comparisons for. If `None`, comparisons are computed for all regressors in the model object. Acceptable values depend on the variable type. See the examples below.
    * Dictionary: keys identify the subset of variables of interest, and values define the type of contrast to compute. Acceptable values depend on the variable type:
    - Categorical variables:
        * "reference": Each factor level is compared to the factor reference (base) level
        * "all": All combinations of observed levels
        * "sequential": Each factor level is compared to the previous factor level
        * "pairwise": Each factor level is compared to all other levels
        * "minmax": The highest and lowest levels of a factor.
        * "revpairwise", "revreference", "revsequential": inverse of the corresponding hypotheses.
        * Vector of length 2 with the two values to compare.
    - Boolean variables:
        * `None`: contrast between True and False
    - Numeric variables:
        * Numeric of length 1: Contrast for a gap of `x`, computed at the observed value plus and minus `x / 2`. For example, estimating a `+1` contrast compares adjusted predictions when the regressor is equal to its observed value minus 0.5 and its observed value plus 0.5.
        * Numeric of length equal to the number of rows in `newdata`: Same as above, but the contrast can be customized for each row of `newdata`.
        * Numeric vector of length 2: Contrast between the 2nd element and the 1st element of the `x` vector.
        * Data frame with the same number of rows as `newdata`, with two columns of "low" and "high" values to compare.
        * Function which accepts a numeric vector and returns a data frame with two columns of "low" and "high" values to compare. See examples below.
        * "iqr": Contrast across the interquartile range of the regressor.
        * "sd": Contrast across one standard deviation around the regressor mean.
        * "2sd": Contrast across two standard deviations around the regressor mean.
        * "minmax": Contrast between the maximum and the minimum values of the regressor.
    - Examples:
        + `variables = {"gear" = "pairwise", "hp" = 10}`
        + `variables = {"gear" = "sequential", "hp" = [100, 120]}`
    - newdata (polars or pandas DataFrame, or str): a data frame or a string specifying where statistics are evaluated in the predictor space. If `None`, unit-level contrasts are computed for each observed value in the original dataset (empirical distribution).
    - comparison (str): a string specifying how pairs of predictions should be compared. See the Comparisons section below for definitions of each transformation.
    - transform (function): a function specifying a transformation applied to unit-level estimates and confidence intervals just before the function returns results. Functions must accept a full column (series) of a Polars data frame and return a corresponding series of the same length. Ex:
        - `transform = numpy.exp`
        - `transform = lambda x: x.exp()`
        - `transform = lambda x: x.map_elements()`
    - equivalence (list): a list of 2 numeric values specifying the bounds used for the two-one-sided test (TOST) of equivalence, and for the non-inferiority and non-superiority tests. See the Details section below.
    - by (bool, str): a logical value, a list of column names in `newdata`. If `True`, estimates are aggregated for each term.
    - hypothesis (str, numpy array): a string specifying a numeric value specifying the null hypothesis used for computing p-values.
    - conf.level (float): a numeric value specifying the confidence level for the confidence intervals. Default is 0.95.

    # Returns:

    The functions return a data.frame with the following columns:

    - term: the name of the variable.
    - contrast: the comparison method used.
    - estimate: the estimated contrast, difference, ratio, or other transformation between pairs of predictions.
    - std_error: the standard error of the estimate.
    - statistic: the test statistic (estimate / std.error).
    - p_value: the p-value of the test.
    - s_value: Shannon transform of the p value.
    - conf_low: the lower confidence interval bound.
    - conf_high: the upper confidence interval bound.

    # Details:

    The `equivalence` argument specifies the bounds used for the two-one-sided test (TOST) of equivalence, and for the non-inferiority and non-superiority tests. The first element specifies the lower bound, and the second element specifies the upper bound. If `None`, equivalence tests are not performed.
    """

    if callable(newdata):
        newdata = newdata(model)

    model = sanitize_model(model)
    by = sanitize_by(by)
    V = sanitize_vcov(vcov, model)
    newdata = sanitize_newdata(model, newdata=newdata, wts=wts, by=by)
    modeldata = model.modeldata
    hypothesis_null = sanitize_hypothesis_null(hypothesis)

    # after sanitize_newdata()
    variables = sanitize_variables(
        variables=variables,
        model=model,
        newdata=newdata,
        comparison=comparison,
        eps=eps,
        by=by,
        wts=wts,
    )

    # pad for character/categorical variables in patsy
    pad = []
    hi = []
    lo = []
    nd = []
    for v in variables:
        nd.append(
            newdata.with_columns(
                pl.lit(v.variable).alias("term"),
                pl.lit(v.lab).alias("contrast"),
                pl.lit(v.comparison).alias("marginaleffects_comparison"),
            )
        )
        hi.append(
            newdata.with_columns(
                pl.lit(v.hi).alias(v.variable),
                pl.lit(v.variable).alias("term"),
                pl.lit(v.lab).alias("contrast"),
                pl.lit(v.comparison).alias("marginaleffects_comparison"),
            )
        )
        lo.append(
            newdata.with_columns(
                pl.lit(v.lo).alias(v.variable),
                pl.lit(v.variable).alias("term"),
                pl.lit(v.lab).alias("contrast"),
                pl.lit(v.comparison).alias("marginaleffects_comparison"),
            )
        )

    # we must pad with *all* variables in the model, not just the ones in the `variables` argument
    vars = model.get_variables_names(variables=None, newdata=modeldata)
    vars = [re.sub(r"\[.*", "", x) for x in vars]
    vars = list(set(vars))
    for v in vars:
        if v in modeldata.columns:
            if modeldata[v].dtype in [pl.Utf8, pl.Boolean]:
                pad.append(get_pad(newdata, v, modeldata[v].unique()))

    # ugly hack, but polars is very strict and `value / 2`` is float
    nd = upcast(nd)
    hi = upcast(hi)
    lo = upcast(lo)
    pad = upcast(pad)
    nd = pl.concat(nd, how="vertical_relaxed")
    hi = pl.concat(hi, how="vertical_relaxed")
    lo = pl.concat(lo, how="vertical_relaxed")
    pad = [x for x in pad if x is not None]
    if len(pad) == 0:
        pad = pl.DataFrame()
    else:
        pad = pl.concat(pad).unique()
        nd = pl.concat(upcast([pad, nd]), how="diagonal")
        hi = pl.concat(upcast([pad, hi]), how="diagonal")
        lo = pl.concat(upcast([pad, lo]), how="diagonal")

    # predictors
    # we want this to be a model matrix to avoid converting data frames to
    # matrices many times, which would be computationally wasteful. But in the
    # case of PyFixest, the predict method only accepts a data frame.
    if isinstance(model, ModelPyfixest):
        hi_X = hi
        lo_X = lo
        nd_X = nd
    else:
        y, hi_X = patsy.dmatrices(model.formula, hi.to_pandas())
        y, lo_X = patsy.dmatrices(model.formula, lo.to_pandas())
        y, nd_X = patsy.dmatrices(model.formula, nd.to_pandas())

    # unpad
    if pad.shape[0] > 0:
        nd_X = nd_X[pad.shape[0] :]
        hi_X = hi_X[pad.shape[0] :]
        lo_X = lo_X[pad.shape[0] :]
        nd = nd[pad.shape[0] :]
        hi = hi[pad.shape[0] :]
        lo = lo[pad.shape[0] :]

    def inner(coefs, by, hypothesis, wts, nd):
        if hasattr(coefs, "to_numpy"):
            coefs = coefs.to_numpy()

        # estimates
        tmp = [
            model.get_predict(params=coefs, newdata=nd_X).rename(
                {"estimate": "predicted"}
            ),
            model.get_predict(params=coefs, newdata=lo_X)
            .rename({"estimate": "predicted_lo"})
            .select("predicted_lo"),
            model.get_predict(params=coefs, newdata=hi_X)
            .rename({"estimate": "predicted_hi"})
            .select("predicted_hi"),
        ]
        tmp = reduce(lambda x, y: pl.concat([x, y], how="horizontal"), tmp)
        if "rowid" in nd.columns and tmp.shape[0] == nd.shape[0]:
            tmp = tmp.with_columns(nd["rowid"].alias("rowid"))

        # no group
        if tmp.shape[0] == nd.shape[0]:
            cols = [x for x in nd.columns if x not in tmp.columns]
            tmp = pl.concat([tmp, nd.select(cols)], how="horizontal")

        # group
        elif "group" in tmp.columns:
            meta = nd.join(tmp.select("group").unique(), how="cross")
            cols = [x for x in meta.columns if x in tmp.columns]
            tmp = meta.join(tmp, on=cols, how="left")

        # not sure what happens here
        else:
            raise ValueError("Something went wrong")

        if isinstance(by, str):
            by = ["term", "contrast"] + [by]
        elif isinstance(by, list):
            by = ["term", "contrast"] + by
        else:
            by = ["term", "contrast"]

        # TODO: problem is that `cyl` is the modified hi and lo instead of the original
        # so when we group by it, we get only rows with cyl matching the contrast.
        def applyfun(x, by, wts=None):
            comp = x["marginaleffects_comparison"][0]
            xvar = x[x["term"][0]]
            if wts is not None:
                xwts = x[wts]
            else:
                xwts = None
            est = estimands[comp](
                hi=x["predicted_hi"],
                lo=x["predicted_lo"],
                eps=eps,
                x=xvar,
                y=x["predicted"],
                w=xwts,
            )
            if est.shape[0] == 1:
                est = est.item()
                tmp = x.select(by).unique().with_columns(pl.lit(est).alias("estimate"))
            else:
                tmp = x.with_columns(pl.lit(est).alias("estimate"))
            return tmp

        def applyfun_outer(x):
            return applyfun(x, by=by, wts=wts)

        # maintain_order is extremely important
        by = [x for x in by if x in tmp.columns]
        tmp = tmp.group_by(by, maintain_order=True).map_groups(applyfun_outer)

        tmp = get_hypothesis(tmp, hypothesis=hypothesis)

        return tmp

    def outer(x):
        return inner(x, by=by, hypothesis=hypothesis, wts=wts, nd=nd)

    out = outer(model.coef)

    if vcov is not None and vcov is not False:
        J = get_jacobian(func=outer, coefs=model.coef, eps_vcov=eps_vcov)
        se = get_se(J, V)
        out = out.with_columns(pl.Series(se).alias("std_error"))
        out = get_z_p_ci(
            out, model, conf_level=conf_level, hypothesis_null=hypothesis_null
        )

    out = get_transform(out, transform=transform)
    out = get_equivalence(out, equivalence=equivalence, df=np.inf)
    out = sort_columns(out, by=by, newdata=newdata)

    out = MarginaleffectsDataFrame(out, by=by, conf_level=conf_level, newdata=newdata)
    return out


def avg_comparisons(
    model,
    variables=None,
    newdata=None,
    comparison="difference",
    vcov=True,
    conf_level=0.95,
    by=True,
    wts=None,
    hypothesis=None,
    equivalence=None,
    transform=None,
    eps=1e-4,
):
    if callable(newdata):
        newdata = newdata(model)

    out = comparisons(
        model=model,
        variables=variables,
        newdata=newdata,
        comparison=comparison,
        vcov=vcov,
        conf_level=conf_level,
        by=by,
        wts=wts,
        hypothesis=hypothesis,
        equivalence=equivalence,
        transform=transform,
        eps=eps,
    )

    return out
