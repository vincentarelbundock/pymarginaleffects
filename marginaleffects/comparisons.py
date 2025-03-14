from .model_pyfixest import ModelPyfixest
from .model_linearmodels import ModelLinearmodels
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
from .utils import get_pad, sort_columns, upcast, ingest
from .docs import docstring_returns

import re
from functools import reduce

import numpy as np
import patsy
import polars as pl


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
    if callable(newdata):
        newdata = newdata(model)

    model = sanitize_model(model)
    by = sanitize_by(by)
    V = sanitize_vcov(vcov, model)
    newdata = sanitize_newdata(model, newdata=newdata, wts=wts, by=by)
    modeldata = model.data
    hypothesis_null = sanitize_hypothesis_null(hypothesis)

    # For each variable in `variables`, this will return two values that we want
    # to compare in the contrast. For example, if there's a variable called
    # "treatment", `sanitize_variables()` may return two values: "lo" vs. "hi", or 0 vs. 1.
    # Important: place after sanitize_newdata()
    variables = sanitize_variables(
        variables=variables,
        model=model,
        newdata=newdata,
        comparison=comparison,
        eps=eps,
        by=by,
        wts=wts,
    )

    # We create two versions of the `newdata` data frame: one where the
    # treatment variable is set to a `hi` value, and one where the treatment is
    # set to a `lo` value.
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

    # Hack: We run into Patsy-related issues unless we "pad" the
    # character/categorical variables to include all unique levels. We add them
    # here but drop them after creating the design matrices.
    vars = model.find_variables()
    vars = [re.sub(r"\[.*", "", x) for x in vars]
    vars = list(set(vars))
    for v in vars:
        if v in modeldata.columns:
            if model.variables_type[v] not in ["numeric", "integer"]:
                pad.append(get_pad(newdata, v, modeldata[v].unique()))

    # Hack: Polars is very strict and `value / 2`` is float, so we need to upcast.
    nd = upcast(nd)
    hi = upcast(hi)
    lo = upcast(lo)
    pad = upcast(pad)

    # nd, hi, and lo are lists of data frames, since the user could have
    # requested many contrasts at the same time using the `variables` argument.
    # We could make predictions on each of them separately, but it's more
    # efficient to combine the data frames and call `predict()` method only
    # once.
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

    # Use the `nd`, `lo`, and `hi` to make counterfactual predictions.
    # data frame -> Patsy -> design matrix -> predict()
    # It is expensive to convert data frames to design matrices, so we do it
    # only once and re-use the design matrices. Unfortunately, this is not
    # possible for PyFixest, since the `.predict()` method it supplies does not
    # accept matrices. So we special-case PyFixest.`
    if isinstance(model, (ModelPyfixest, ModelLinearmodels)):
        hi_X = hi
        lo_X = lo
        nd_X = nd
    else:
        y, hi_X = patsy.dmatrices(model.formula, ingest(hi).to_pandas())
        y, lo_X = patsy.dmatrices(model.formula, ingest(lo).to_pandas())
        y, nd_X = patsy.dmatrices(model.formula, ingest(nd).to_pandas())

    # unpad
    if pad.shape[0] > 0:
        nd_X = nd_X[pad.shape[0] :]
        hi_X = hi_X[pad.shape[0] :]
        lo_X = lo_X[pad.shape[0] :]
        nd = nd[pad.shape[0] :]
        hi = hi[pad.shape[0] :]
        lo = lo[pad.shape[0] :]

    # inner() takes the `hi` and `lo` matrices, computes predictions, compares
    # them, and aggregates the results based on the `by` argument. This gives us
    # the final quantity of interest. We wrap this in a function because it will
    # be called multiple times with slightly different values of the `coefs`.
    # This is necessary to compute the numerical derivatives in the Jacobian
    # that we use to compute standard errors, where individual entries are
    # derivatives of a contrast with respect to one of the model coefficients.
    def inner(coefs, by, hypothesis, wts, nd):
        if hasattr(coefs, "to_numpy"):
            coefs = coefs.to_numpy()

        # main unit-level estimates
        tmp = [
            # fitted values
            model.get_predict(params=coefs, newdata=nd_X).rename(
                {"estimate": "predicted"}
            ),
            # predictions for the "lo" counterfactual
            model.get_predict(params=coefs, newdata=lo_X)
            .rename({"estimate": "predicted_lo"})
            .select("predicted_lo"),
            # predictions for the "hi" counterfactual
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

        # group (categorical outcome models)
        elif "group" in tmp.columns:
            meta = nd.join(tmp.select("group").unique(), how="cross")
            cols = [x for x in meta.columns if x in tmp.columns]
            tmp = meta.join(tmp, on=cols, how="left")

        # not sure what happens here
        else:
            raise ValueError("Something went wrong")

        # column names on which we will aggregate results
        if isinstance(by, str):
            by = ["term", "contrast"] + [by]
        elif isinstance(by, list):
            by = ["term", "contrast"] + by
        else:
            by = ["term", "contrast"]

        # apply a function to compare the predicted_hi and predicted_lo columns
        # ex: hi-lo, mean(hi-lo), hi/lo, mean(hi)/mean(lo), etc.
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

        # maintain_order is extremely important
        by = [x for x in by if x in tmp.columns]
        tmp = tmp.group_by(*by, maintain_order=True).map_groups(
            function=lambda x: applyfun(x, by=by, wts=wts)
        )

        tmp = get_hypothesis(tmp, hypothesis=hypothesis, by=by)

        return tmp

    # outer() is a wrapper with a single argument `x`, the model coefficients.
    # Just for convenience when taking derivatives with respect to the
    # coefficients.
    def outer(x):
        return inner(x, by=by, hypothesis=hypothesis, wts=wts, nd=nd)

    out = outer(model.coef)

    # Compute standard errors and confidence intervals
    if vcov is not None and vcov is not False and V is not None:
        J = get_jacobian(func=outer, coefs=model.coef, eps_vcov=eps_vcov)
        se = get_se(J, V)
        out = out.with_columns(pl.Series(se).alias("std_error"))
        out = get_z_p_ci(
            out, model, conf_level=conf_level, hypothesis_null=hypothesis_null
        )
    else:
        J = None

    # Apply a few final operations
    out = get_transform(out, transform=transform)
    out = get_equivalence(out, equivalence=equivalence, df=np.inf)
    out = sort_columns(out, by=by, newdata=newdata)

    # Wrap things up in a nice class
    out = MarginaleffectsDataFrame(
        out, by=by, conf_level=conf_level, jacobian=J, newdata=newdata
    )
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
    """
    docstring (TO DO)
    """
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


docs_comparisons = """

# `comparisons()`

`comparisons()` and `avg_comparisons()` are functions for predicting the outcome variable at different regressor values and comparing those predictions by computing a difference, ratio, or some other function. These functions can return many quantities of interest, such as contrasts, differences, risk ratios, changes in log odds, lift, slopes, elasticities, etc.

## Parameters

* model : object Model object fitted using the `statsmodels` formula API.
* variables : str, list, dictionary
    - a string, list of strings, or dictionary of variables to compute comparisons for. If `None`, comparisons are computed for all regressors in the model object. Acceptable values depend on the variable type. See the examples below.
    - Dictionary: keys identify the subset of variables of interest, and values define the type of contrast to compute. Acceptable values depend on the variable type:
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
* newdata : polars or pandas DataFrame, or str
    - Data frame or string specifying where statistics are evaluated in the predictor space. If `None`, unit-level contrasts are computed for each observed value in the original dataset (empirical distribution).
* comparison : str
    - String specifying how pairs of predictions should be compared. See the Comparisons section below for definitions of each transformation.
* transform : function
    Function specifying a transformation applied to unit-level estimates and confidence intervals just before the function returns results. Functions must accept a full column (series) of a Polars data frame and return a corresponding series of the same length. Ex:
        - `transform = numpy.exp`
        - `transform = lambda x: x.exp()`
        - `transform = lambda x: x.map_elements()`
* equivalence : list
    - List of 2 numeric values specifying the bounds used for the two-one-sided test (TOST) of equivalence, and for the non-inferiority and non-superiority tests. See the Details section below.
* by : bool, str
    - Logical value, list of column names in `newdata`. If `True`, estimates are aggregated for each term.
* hypothesis : str, numpy array
    - String specifying a numeric value specifying the null hypothesis used for computing p-values.
* conf_level : float
    - Numeric value specifying the confidence level for the confidence intervals. Default is 0.95.

# Examples


# Details

The `equivalence` argument specifies the bounds used for the two-one-sided test (TOST) of equivalence, and for the non-inferiority and non-superiority tests. The first element specifies the lower bound, and the second element specifies the upper bound. If `None`, equivalence tests are not performed.
"""


comparisons.__doc__ = docs_comparisons + docstring_returns

avg_comparisons.__doc__ = comparisons.__doc__
