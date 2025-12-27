import numpy as np
import polars as pl

from .by import get_by
from .result import MarginaleffectsResult
from .equivalence import get_equivalence
from .hypothesis import get_hypothesis
from .sanitize_model import sanitize_model
from .sanity import (
    sanitize_by,
    sanitize_hypothesis_null,
    sanitize_newdata,
    sanitize_vcov,
)
from .transform import get_transform
from .uncertainty import get_jacobian, get_se, get_z_p_ci
from .utils import sort_columns, validate_string_columns
from .model_pyfixest import ModelPyfixest
from .model_linearmodels import ModelLinearmodels
from .formulaic_utils import model_matrices
from warnings import warn

from .docs import (
    DocsDetails,
    DocsParameters,
    docstring_returns,
)


def predictions(
    model,
    variables=None,
    conf_level=0.95,
    vcov=True,
    by=False,
    newdata=None,
    hypothesis=None,
    equivalence=None,
    transform=None,
    wts=None,
    eps_vcov=None,
    **kwargs,
):
    """
    `predictions()` and `avg_predictions()` predict outcomes using a fitted model on a specified scale for given combinations of values of predictor variables, such as their observed values, means, or factor levels (reference grid).

    For more information, visit the website: https://marginaleffects.com/

    Or type: `help(predictions)`
    """
    if "hypotheses" in kwargs:
        if hypothesis is not None:
            raise ValueError("Specify at most one of `hypothesis` or `hypotheses`.")
        hypotheses = kwargs.pop("hypotheses")
        warn(
            "`hypotheses` is deprecated; use `hypothesis` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        hypothesis = hypotheses
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise TypeError(
            f"predictions() got unexpected keyword argument(s): {unexpected}"
        )

    if callable(newdata):
        newdata = newdata(model)

    # sanity checks
    model = sanitize_model(model)

    # For pyfixest models, automatically set vcov=False for predictions with warning
    if isinstance(model, ModelPyfixest) and vcov is not False:
        has_fixef = getattr(model.model, "_has_fixef", False)
        if has_fixef:
            warn(
                "For this pyfixest model, marginaleffects cannot take into account the "
                "uncertainty in fixed-effects parameters. Standard errors are disabled "
                "and vcov=False is enforced.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warn(
                "Standard errors are not available for predictions in pyfixest models. "
                "Setting vcov=False automatically.",
                UserWarning,
                stacklevel=2,
            )
        vcov = False

    by = sanitize_by(by)
    V = sanitize_vcov(vcov, model)
    newdata = sanitize_newdata(model, newdata, wts=wts, by=by)
    hypothesis_null = sanitize_hypothesis_null(hypothesis)

    modeldata = model.get_modeldata()

    # Validate that columns used in by are not String type
    validate_string_columns(by, modeldata, context="the 'by' parameter")

    if variables:
        # convert to dictionary
        if isinstance(variables, str):
            variables = {variables: None}
        elif isinstance(variables, list):
            for v in variables:
                if not isinstance(v, str):
                    raise TypeError(
                        "All entries in the `variables` list must be strings."
                    )
            variables = {v: None for v in variables}
        elif not isinstance(variables, dict):
            raise TypeError("`variables` argument must be a dictionary")

        for variable, value in variables.items():
            if callable(value):
                val = value()
            elif value is None:
                val = modeldata[variable].unique()
            elif value == "sd":
                std = modeldata[variable].std()
                mean = modeldata[variable].mean()
                val = [mean - std / 2, mean + std / 2]
            elif value == "2sd":
                std = modeldata[variable].std()
                mean = modeldata[variable].mean()
                val = [mean - std, mean + std]
            elif value == "iqr":
                val = [
                    np.percentile(newdata[variable], 75),
                    np.percentile(newdata[variable], 25),
                ]
            elif value == "minmax":
                val = [np.max(newdata[variable]), np.min(newdata[variable])]
            elif value == "threenum":
                std = modeldata[variable].std()
                mean = modeldata[variable].mean()
                val = [mean - std / 2, mean, mean + std / 2]
            elif value == "fivenum":
                val = np.percentile(
                    modeldata[variable], [0, 25, 50, 75, 100], method="midpoint"
                )
            else:
                val = value

            newdata = newdata.drop(variable)
            newdata = newdata.join(pl.DataFrame({variable: val}), how="cross")
            newdata = newdata.sort(variable)

        newdata.datagrid_explicit = list(variables.keys())

    # predictors
    # we want this to be a model matrix to avoid converting data frames to
    # matrices many times, which would be computationally wasteful. But in the
    # case of PyFixest, the predict method only accepts a data frame.
    if isinstance(model, ModelPyfixest):
        exog = newdata.to_pandas()
    # Linearmodels accepts polars dataframes and converts them to Pandas internally
    elif isinstance(model, ModelLinearmodels):
        exog = newdata
    else:
        if hasattr(model, "design_info_patsy"):
            f = model.design_info_patsy
        else:
            f = model.get_formula()

        if callable(f):
            endog, exog = f(newdata)

        else:
            endog, exog = model_matrices(
                f, newdata, formula_engine=model.get_formula_engine()
            )

    # estimands
    def inner(x):
        out = model.get_predict(params=np.array(x), newdata=exog)

        if out.shape[0] == newdata.shape[0]:
            cols = [x for x in newdata.columns if x not in out.columns]
            out = pl.concat([out, newdata.select(cols)], how="horizontal")

        # group
        elif "group" in out.columns:
            meta = newdata.join(out.select("group").unique(), how="cross")
            cols = [x for x in meta.columns if x in out.columns]
            out = meta.join(out, on=cols, how="left")

        # not sure what happens here
        else:
            raise ValueError("Something went wrong")

        out = get_by(model, out, newdata=newdata, by=by, wts=wts)
        out = get_hypothesis(out, hypothesis=hypothesis, by=by)
        return out

    out = inner(model.get_coef())

    if V is not None:
        J = get_jacobian(inner, model.get_coef(), eps_vcov=eps_vcov)
        se = get_se(J, V)
        out = out.with_columns(pl.Series(se).alias("std_error"))
        out = get_z_p_ci(
            out, model, conf_level=conf_level, hypothesis_null=hypothesis_null
        )
    else:
        J = None
    out = get_transform(out, transform=transform)
    out = get_equivalence(out, equivalence=equivalence)
    out = sort_columns(out, by=by, newdata=newdata)

    out = MarginaleffectsResult(
        out, by=by, conf_level=conf_level, jacobian=J, newdata=newdata
    )
    return out


def avg_predictions(
    model,
    variables=None,
    conf_level=0.95,
    vcov=True,
    by=True,
    newdata=None,
    hypothesis=None,
    equivalence=None,
    transform=None,
    wts=None,
    **kwargs,
):
    """
    `predictions()` and `avg_predictions()` predict outcomes using a fitted model on a specified scale for given combinations of values of predictor variables, such as their observed values, means, or factor levels (reference grid).

    For more information, visit the website: https://marginaleffects.com/

    Or type: `help(avg_predictions)`
    """
    if callable(newdata):
        newdata = newdata(model)

    out = predictions(
        model=model,
        variables=variables,
        conf_level=conf_level,
        vcov=vcov,
        by=by,
        newdata=newdata,
        hypothesis=hypothesis,
        equivalence=equivalence,
        transform=transform,
        wts=wts,
        **kwargs,
    )

    return out


docs_predictions = (
    """
# `predictions()`

`predictions()` and `avg_predictions()` predict outcomes using a fitted model on a specified scale for given combinations of values of predictor variables, such as their observed values, means, or factor levels (reference grid).
    
* `predictions()`: unit-level (conditional) estimates.
* `avg_predictions()`: average (marginal) estimates.

See the package website and vignette for examples:

- https://marginaleffects.com/chapters/predictions.html
- https://marginaleffects.com

## Parameters
"""
    + DocsParameters.docstring_model
    + DocsParameters.docstring_variables("prediction")
    + DocsParameters.docstring_newdata("prediction")
    + DocsParameters.docstring_by
    + DocsParameters.docstring_transform
    + DocsParameters.docstring_hypothesis
    + DocsParameters.docstring_wts
    + DocsParameters.docstring_vcov
    + DocsParameters.docstring_equivalence
    + DocsParameters.docstring_conf_level
    + DocsParameters.docstring_eps_vcov
    + docstring_returns
    + """ 
## Examples
```py
from marginaleffects import *

import statsmodels.api as sm
import statsmodels.formula.api as smf
data = get_dataset("thornton")

mod = smf.ols("outcome ~ incentive + distance", data).fit()

predictions(mod)

avg_predictions(mod)

predictions(mod, by = "village")

avg_predictions(mod, by = "village")

predictions(mod, hypothesis = 3)

avg_predictions(mod, hypothesis = 3)
```

## Details
"""
    + DocsDetails.docstring_tost
    + DocsDetails.docstring_order_of_operations
)


predictions.__doc__ = docs_predictions

avg_predictions.__doc__ = predictions.__doc__
