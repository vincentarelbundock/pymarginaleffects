import numpy as np
import patsy
import polars as pl

from .by import get_by
from .equivalence import get_equivalence
from .hypothesis import get_hypothesis
from .sanity import sanitize_newdata, sanitize_vcov, sanitize_by, sanitize_hypothesis_null
from .transform import get_transform
from .uncertainty import get_jacobian, get_se, get_z_p_ci
from .utils import sort_columns, get_pad, upcast
from .getters import get_modeldata, get_variables_names, get_predict, get_coef
from .classes import MarginaleffectsDataFrame




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
):
    """
    Predict outcomes using a fitted model on a specified scale for given combinations of values
    of predictor variables, such as their observed values, means, or factor levels (reference grid).

    This function handles unit-level (conditional) estimates and average (marginal) estimates based
    on the `variables` and `newdata` arguments. See the package website and vignette for examples:
    - https://vincentarelbundock.github.io/marginaleffects/articles/predictions.html
    - https://vincentarelbundock.github.io/marginaleffects/

    Parameters
    ----------
    model : object
        Model object.

    newdata : Union[None, DataFrame], optional
        Grid of predictor values at which to evaluate predictions, by default predictions are made on the data used to fit the model.

    by (bool, str): a logical value, a list of column names in `newdata`. If `True`, estimates are aggregated for each term.

    wts: Column name of weights to use for marginalization. Must be a column in `newdata`

    transform : Callable, optional
        A function applied to unit-level adjusted predictions and confidence intervals just before
        the function returns results, by default None.

    hypothesis: String formula of hypothesis test or numpy array.

    Returns
    -------
    DataFrame
        A DataFrame with one row per observation and several columns:
        - rowid: row number of the `newdata` data frame
        - type: prediction type, as defined by the `type` argument
        - group: (optional) value of the grouped outcome (e.g., categorical outcome models)
        - estimate: predicted outcome
        - std.error: standard errors computed using the delta method.
        - p_value: p value associated with the `estimate` column.
        - s_value: Shannon information transforms of p values.
        - conf_low: lower bound of the confidence interval (or equal-tailed interval for Bayesian models)
        - conf_high: upper bound of the confidence interval (or equal-tailed interval for Bayesian models)
    """

    # sanity checks
    by = sanitize_by(by)
    V = sanitize_vcov(vcov, model)
    newdata = sanitize_newdata(model, newdata, wts=wts, by=by)
    hypothesis_null = sanitize_hypothesis_null(hypothesis)

    modeldata = get_modeldata(model)

    if variables:
        if not isinstance(variables, dict):
            raise TypeError("`variables` argument must be a dictionary")
        for variable, value in variables.items():
            if callable(value):
                val = value()
            elif value == "sd":
                std = modeldata[variable].std()
                mean = modeldata[variable].mean()
                val = [mean - std / 2, mean + std / 2]
            elif value == "2sd":
                std = modeldata[variable].std()
                mean = modeldata[variable].mean()
                val = [mean - std, mean + std]
            elif value == "iqr":
                val = [np.percentile(newdata[variable], 75), np.percentile(newdata[variable], 25)]
            elif value == "minmax":
                val = [np.max(newdata[variable]), np.min(newdata[variable])]
            elif value == "threenum":
                std = modeldata[variable].std()
                mean = modeldata[variable].mean()
                val = [mean - std / 2, mean, mean + std / 2]
            elif value == "fivenum":
                val = np.percentile(modeldata[variable], [0, 25, 50, 75, 100], method="midpoint")
            else:
                val = value

            newdata = newdata.drop(variable)
            newdata = newdata.join(pl.DataFrame({variable:val}), how = "cross")
            newdata = newdata.sort(variable)

        newdata.datagrid_explicit = list(variables.keys())

    # pad
    pad = []
    vs = get_variables_names(variables = None, model = model, newdata = modeldata)
    for v in vs:
        if not newdata[v].is_numeric():
            uniqs = modeldata[v].unique()
            if not all(uniq in newdata[v] for uniq in uniqs):
                pad.append(get_pad(modeldata, v, uniqs))
    if len(pad) > 0:
        pad = pl.concat(pad)
        tmp = upcast([newdata, pad])
        newdata = pl.concat(tmp, how = "diagonal")
    else:
        pad = pl.DataFrame()

    # predictors
    y, exog = patsy.dmatrices(model.model.formula, newdata.to_pandas())

    # estimands
    def inner(x):
        out = get_predict(model, np.array(x), exog)

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
        out = get_hypothesis(out, hypothesis=hypothesis)
        return out

    out = inner(get_coef(model))

    if vcov is not None:
        J = get_jacobian(inner, get_coef(model))
        se = get_se(J, V)
        out = out.with_columns(pl.Series(se).alias("std_error"))
        out = get_z_p_ci(out, model, conf_level=conf_level, hypothesis_null=hypothesis_null)
    out = get_transform(out, transform=transform)
    out = get_equivalence(out, equivalence=equivalence)
    out = sort_columns(out, by=by, newdata=newdata)

    # unpad
    if "rowid" in out.columns and pad.shape[0] > 0:
        out = out[:-pad.shape[0]:]

    out = MarginaleffectsDataFrame(out, by=by, conf_level=conf_level, newdata=newdata)
    return out


def avg_predictions(
    model,
    conf_level=0.95,
    vcov=True,
    by=True,
    newdata=None,
    hypothesis=None,
    equivalence=None,
    transform=None,
    wts=None,
):
    out = predictions(
        model=model,
        conf_level=conf_level,
        vcov=vcov,
        by=by,
        newdata=newdata,
        hypothesis=hypothesis,
        equivalence=equivalence,
        transform=transform,
        wts=wts,
    )

    return out
