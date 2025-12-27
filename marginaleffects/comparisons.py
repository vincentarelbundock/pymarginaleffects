import re
from functools import reduce

import numpy as np
import patsy
import polars as pl

from .result import MarginaleffectsResult
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
    handle_deprecated_hypotheses_argument,
    handle_pyfixest_vcov_limitation,
)
from .transform import get_transform
from .uncertainty import get_jacobian, get_se, get_z_p_ci
from .utils import get_pad, sort_columns, upcast, validate_string_columns
from .model_pyfixest import ModelPyfixest
from .model_sklearn import ModelSklearn
from .model_linearmodels import ModelLinearmodels

from .docs import (
    DocsDetails,
    DocsParameters,
    docstring_returns,
)


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
    cross=False,
    transform=None,
    eps=1e-4,
    eps_vcov=None,
    **kwargs,
):
    """
    `comparisons()` and `avg_comparisons()` are functions for predicting the outcome variable at different regressor values and comparing those predictions by computing a difference, ratio, or some other function. These functions can return many quantities of interest, such as contrasts, differences, risk ratios, changes in log odds, lift, slopes, elasticities, average treatment effect (on the treated or untreated), etc.

    For more information, visit the website: https://marginaleffects.com/

    Or type: `help(comparisons)`
    """
    hypothesis = handle_deprecated_hypotheses_argument(hypothesis, kwargs, stacklevel=2)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(
            f"comparisons() got unexpected keyword argument(s): {unexpected}"
        )

    if callable(newdata):
        newdata = newdata(model)

    model = sanitize_model(model)
    vcov = handle_pyfixest_vcov_limitation(model, vcov, stacklevel=2)

    by = sanitize_by(by)
    V = sanitize_vcov(vcov, model)
    newdata = sanitize_newdata(model, newdata=newdata, wts=wts, by=by)
    modeldata = model.get_modeldata()
    hypothesis_null = sanitize_hypothesis_null(hypothesis)

    # Validate cross parameter
    if cross and variables is None:
        raise ValueError(
            "The `variables` argument must be specified when `cross=True`."
        )

    # Validate that columns used in by and variables are not String type
    validate_string_columns(by, modeldata, context="the 'by' parameter")
    validate_string_columns(variables, modeldata, context="the 'variables' parameter")

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
        cross=cross,
    )

    # We create two versions of the `newdata` data frame: one where the
    # treatment variable is set to a `hi` value, and one where the treatment is
    # set to a `lo` value.
    pad = []
    hi = []
    lo = []
    nd = []
    if not cross:
        for v in variables:
            if callable(v.comparison):
                vcomp = "custom"
            else:
                vcomp = v.comparison
            nd.append(
                newdata.with_columns(
                    pl.lit(v.variable).alias("term"),
                    pl.lit(v.lab).alias("contrast"),
                    pl.lit(vcomp).alias("marginaleffects_comparison"),
                )
            )
            hi.append(
                newdata.with_columns(
                    pl.lit(v.hi).alias(v.variable),
                    pl.lit(v.variable).alias("term"),
                    pl.lit(v.lab).alias("contrast"),
                    pl.lit(vcomp).alias("marginaleffects_comparison"),
                )
            )
            lo.append(
                newdata.with_columns(
                    pl.lit(v.lo).alias(v.variable),
                    pl.lit(v.variable).alias("term"),
                    pl.lit(v.lab).alias("contrast"),
                    pl.lit(vcomp).alias("marginaleffects_comparison"),
                )
            )

    else:
        # Check if we have factorial grid HiLo objects (variable is a tuple)
        if variables and isinstance(variables[0].variable, tuple):
            # Factorial grid comparisons - process each HiLo independently
            for v in variables:
                if callable(v.comparison):
                    vcomp = "custom"
                else:
                    vcomp = v.comparison

                var_names = v.variable  # Tuple of variable names
                nd_row = newdata.clone()
                hi_row = newdata.clone()
                lo_row = newdata.clone()

                # Set all variables from hi/lo (which contain list of values)
                for k, var_name in enumerate(var_names):
                    hi_row = hi_row.with_columns(pl.lit(v.hi[k]).alias(var_name))
                    lo_row = lo_row.with_columns(pl.lit(v.lo[k]).alias(var_name))

                    # Create contrast labels for each variable
                    contrast_label = f"{v.hi[k]} - {v.lo[k]}"
                    nd_row = nd_row.with_columns(
                        pl.lit(contrast_label).alias(f"contrast_{var_name}")
                    )
                    hi_row = hi_row.with_columns(
                        pl.lit(contrast_label).alias(f"contrast_{var_name}")
                    )
                    lo_row = lo_row.with_columns(
                        pl.lit(contrast_label).alias(f"contrast_{var_name}")
                    )

                # Add metadata
                nd_row = nd_row.with_columns(
                    pl.lit(var_names[0]).alias("term"),
                    pl.lit(vcomp).alias("marginaleffects_comparison"),
                )
                hi_row = hi_row.with_columns(
                    pl.lit(var_names[0]).alias("term"),
                    pl.lit(vcomp).alias("marginaleffects_comparison"),
                )
                lo_row = lo_row.with_columns(
                    pl.lit(var_names[0]).alias("term"),
                    pl.lit(vcomp).alias("marginaleffects_comparison"),
                )

                nd.append(nd_row)
                hi.append(hi_row)
                lo.append(lo_row)
        else:
            # Original cross logic for simple cases
            hi.append(newdata)
            lo.append(newdata)
            nd.append(newdata)
            for i, v in enumerate(variables):
                if callable(v.comparison):
                    vcomp = "custom"
                else:
                    vcomp = v.comparison
                nd[0] = nd[0].with_columns(
                    pl.lit(v.variable).alias("term"),
                    pl.lit(v.lab).alias(f"contrast_{v.variable}"),
                    pl.lit(vcomp).alias("marginaleffects_comparison"),
                )
                hi[0] = hi[0].with_columns(
                    pl.lit(v.hi).alias(v.variable),
                    pl.lit(v.variable).alias("term"),
                    pl.lit(v.lab).alias(f"contrast_{v.variable}"),
                    pl.lit(vcomp).alias("marginaleffects_comparison"),
                )
                lo[0] = lo[0].with_columns(
                    pl.lit(v.lo).alias(v.variable),
                    pl.lit(v.variable).alias("term"),
                    pl.lit(v.lab).alias(f"contrast_{v.variable}"),
                    pl.lit(vcomp).alias("marginaleffects_comparison"),
                )

    # Hack: We run into Patsy-related issues unless we "pad" the
    # character/categorical variables to include all unique levels. We add them
    # here but drop them after creating the design matrices.
    vars = model.find_variables()
    if vars is not None:
        vars = [re.sub(r"\[.*", "", x) for x in vars]
        vars = list(set(vars))
        for v in vars:
            if v in modeldata.columns:
                if model.get_variable_type(v) not in ["numeric", "integer"]:
                    pad.append(get_pad(newdata, v, modeldata[v].unique()))

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
        for i, v in enumerate(pad):
            pad[i] = upcast(v, pad[i - 1])
        pad = pl.concat(pad, how="diagonal").unique()
        pad = pad.with_columns(pl.lit(-1).alias("rowid"))

    # manipulated data must have at least the same precision as the modeldata
    lo = upcast(lo, modeldata)
    hi = upcast(hi, modeldata)
    pad = upcast(pad, modeldata)
    nd = upcast(nd, modeldata)

    # non-manipulated data must have at least the smae precision as manipulated data
    # ex: int + 0.0001 for slopes
    pad = upcast(pad, hi)
    nd = upcast(nd, hi)

    # Handle schema alignment for List columns before concat
    dfs_to_align = [("nd", nd), ("hi", hi), ("lo", lo)]

    for df_name, df in dfs_to_align:
        common_cols = set(pad.columns) & set(df.columns)
        for col in common_cols:
            pad_dtype = str(pad[col].dtype)
            df_dtype = str(df[col].dtype)
            if pad_dtype != df_dtype:
                # Handle List type mismatches
                if pad_dtype.startswith("List(") and df_dtype.startswith("List("):
                    # Both are List types but with different inner types
                    # Convert both to simple string lists to ensure compatibility
                    try:
                        if col in pad.columns:
                            pad = pad.with_columns(
                                pad[col]
                                .list.eval(pl.element().cast(pl.String))
                                .alias(col)
                            )
                        if col in df.columns:
                            df = df.with_columns(
                                df[col]
                                .list.eval(pl.element().cast(pl.String))
                                .alias(col)
                            )
                    except Exception as e:
                        print(
                            f"Warning: Could not convert List column {col} to strings: {e}"
                        )
                        # Fallback: try to explode List columns to regular columns
                        try:
                            if col in pad.columns and pad.height > 0:
                                pad = pad.explode(col)
                            if col in df.columns and df.height > 0:
                                df = df.explode(col)
                        except Exception as e2:
                            print(f"Warning: Could not explode List column {col}: {e2}")
                            # Last resort: convert to string representation
                            if col in pad.columns:
                                pad = pad.with_columns(
                                    pad[col].cast(pl.String).alias(col)
                                )
                            if col in df.columns:
                                df = df.with_columns(df[col].cast(pl.String).alias(col))

        # Update the variable with the aligned DataFrame
        if df_name == "nd":
            nd = df
        elif df_name == "hi":
            hi = df
        elif df_name == "lo":
            lo = df

    nd = pl.concat([pad, nd], how="diagonal")
    hi = pl.concat([pad, hi], how="diagonal")
    lo = pl.concat([pad, lo], how="diagonal")

    # Explode any remaining List columns to avoid issues with pandas/patsy
    # Only explode if we have categorical/string List columns that need to be handled
    list_cols = [col for col in nd.columns if str(nd[col].dtype).startswith("List(")]
    categorical_list_cols = []
    for col in list_cols:
        dtype_str = str(nd[col].dtype)
        # Only explode List columns that contain categorical/string data
        if (
            "Enum(" in dtype_str or "String" in dtype_str or "UInt32" in dtype_str
        ) and col in ["Region"]:
            categorical_list_cols.append(col)

    if categorical_list_cols:
        for col in categorical_list_cols:
            nd = nd.explode(col)
            hi = hi.explode(col)
            lo = lo.explode(col)

    # response cannot be NULL

    # Use the `nd`, `lo`, and `hi` to make counterfactual predictions.
    # data frame -> Patsy -> design matrix -> predict()
    # It is expensive to convert data frames to design matrices, so we do it
    # only once and re-use the design matrices. Unfortunately, this is not
    # possible for PyFixest, since the `.predict()` method it supplies does not
    # accept matrices. So we special-case PyFixest.`
    if isinstance(model, (ModelPyfixest, ModelLinearmodels, ModelSklearn)):
        hi_X = hi
        lo_X = lo
        nd_X = nd
    else:
        fml = re.sub(r".*~", "", model.get_formula())
        hi_X = patsy.dmatrix(fml, hi.to_pandas())
        lo_X = patsy.dmatrix(fml, lo.to_pandas())
        nd_X = patsy.dmatrix(fml, nd.to_pandas())

    # unpad
    if pad.shape[0] >= 0:
        nd_X = nd_X[pad.shape[0] :]
        hi_X = hi_X[pad.shape[0] :]
        lo_X = lo_X[pad.shape[0] :]
        nd = nd[pad.shape[0] :]
        hi = hi[pad.shape[0] :]
        lo = lo[pad.shape[0] :]

    # Create a mapping of comparison labels to their actual functions (for callable comparisons)
    comparison_functions = {}
    for v in variables:
        if callable(v.comparison):
            # Store the callable function for later use, keyed by its index or variable+lab combo
            key = f"{v.variable}_{v.lab}"
            comparison_functions[key] = v.comparison

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

        by = by + [x for x in tmp.columns if x.startswith("contrast_")]

        # apply a function to compare the predicted_hi and predicted_lo columns
        # ex: hi-lo, mean(hi-lo), hi/lo, mean(hi)/mean(lo), etc.
        def applyfun(x, by, wts=None):
            comp = x["marginaleffects_comparison"][0]
            xvar = x[x["term"][0]]
            if wts is not None:
                xwts = x[wts]
            else:
                xwts = None

            # Check if this is a custom callable comparison
            term_val = x["term"][0] if "term" in x.columns else None
            contrast_val = x["contrast"][0] if "contrast" in x.columns else None
            key = f"{term_val}_{contrast_val}"

            if comp == "custom" and key in comparison_functions:
                # Use the callable comparison function
                estimand = comparison_functions[key]
            else:
                # Use the predefined estimand
                estimand = estimands[comp]

            est = estimand(
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

    out = outer(model.get_coef())

    # Compute standard errors and confidence intervals
    if vcov is not None and vcov is not False and V is not None:
        J = get_jacobian(func=outer, coefs=model.get_coef(), eps_vcov=eps_vcov)
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

    # not sure why we can't do this earlier. Might be related to this bug report
    # https://github.com/pola-rs/polars/issues/14401
    if cross:
        out = out.with_columns(pl.lit("cross").alias("term"))

    # Wrap things up in a nice class
    out = MarginaleffectsResult(
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
    cross=False,
    transform=None,
    eps=1e-4,
    **kwargs,
):
    """
    `comparisons()` and `avg_comparisons()` are functions for predicting the outcome variable at different regressor values and comparing those predictions by computing a difference, ratio, or some other function. These functions can return many quantities of interest, such as contrasts, differences, risk ratios, changes in log odds, lift, slopes, elasticities, average treatment effect (on the treated or untreated), etc.

    For more information, visit the website: https://marginaleffects.com/

    Or type: `help(avg_comparisons)`
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
        cross=cross,
        transform=transform,
        eps=eps,
        **kwargs,
    )

    return out


docs_comparisons = (
    """

# `comparisons()`

`comparisons()` and `avg_comparisons()` are functions for predicting the outcome variable at different regressor values and comparing those predictions by computing a difference, ratio, or some other function. These functions can return many quantities of interest, such as contrasts, differences, risk ratios, changes in log odds, lift, slopes, elasticities, average treatment effect (on the treated or untreated), etc.

* `comparisons()`: unit-level (conditional) estimates.
* `avg_comparisons()`: average (marginal) estimates.

See the package website and vignette for examples:

* https://marginaleffects.com/chapters/comparisons.html
* https://marginaleffects.com

## Parameters
"""
    + DocsParameters.docstring_model
    + DocsParameters.docstring_variables("comparison")
    + DocsParameters.docstring_newdata("comparison")
    + """
* `comparison`: (str or callable) String specifying how pairs of predictions should be compared, or a callable function to compute custom estimates. See the Comparisons section below for definitions of each transformation.

  * Acceptable strings: difference, differenceavg, differenceavgwts, dydx, eyex, eydx, dyex, dydxavg, eyexavg, eydxavg, dyexavg, dydxavgwts, eyexavgwts, eydxavgwts, dyexavgwts, ratio, ratioavg, ratioavgwts, lnratio, lnratioavg, lnratioavgwts, lnor, lnoravg, lnoravgwts, lift, liftavg, liftavg, expdydx, expdydxavg, expdydxavgwts

  * Callable: A function that takes `hi`, `lo`, `eps`, `x`, `y`, and `w` as arguments and returns a numeric array. This allows computing custom comparisons like `lambda hi, lo, eps, x, y, w: hi / lo` for ratios or `lambda hi, lo, eps, x, y, w: (hi - lo) / lo * 100` for percent changes.
"""
    + DocsParameters.docstring_by
    + DocsParameters.docstring_transform
    + DocsParameters.docstring_hypothesis
    + DocsParameters.docstring_wts
    + DocsParameters.docstring_vcov
    + DocsParameters.docstring_equivalence
    + DocsParameters.docstring_cross
    + DocsParameters.docstring_conf_level
    + DocsParameters.docstring_eps
    + DocsParameters.docstring_eps_vcov
    + docstring_returns
    + """
## Examples
```py
from marginaleffects import *
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf
data = get_dataset("thornton")
model = smf.ols("outcome ~ distance + incentive", data=data).fit()

# Basic comparisons
comparisons(model)

avg_comparisons(model)

comparisons(model, hypothesis=0)

avg_comparisons(model, hypothesis=0)

comparisons(model, by="agecat")

avg_comparisons(model, by="agecat")

# Custom comparisons with lambda functions
# Ratio comparison using lambda
comparisons(model, variables="distance",
            comparison=lambda hi, lo, eps, x, y, w: hi / lo)

# Percent change using lambda
comparisons(model, variables="distance",
            comparison=lambda hi, lo, eps, x, y, w: (hi - lo) / lo * 100)

# Log ratio using lambda
comparisons(model, variables="distance",
            comparison=lambda hi, lo, eps, x, y, w: np.log(hi / lo))
```

## Details
"""
    + DocsDetails.docstring_tost
    + DocsDetails.docstring_order_of_operations
    + ""  # add comparisons argument functions section as in R at https://marginaleffects.com/man/r/comparisons.html
)


comparisons.__doc__ = docs_comparisons

avg_comparisons.__doc__ = comparisons.__doc__
