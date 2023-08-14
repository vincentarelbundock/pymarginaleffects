import numpy as np
import patsy
import polars as pl

from .by import get_by
from .equivalence import get_equivalence
from .hypothesis import get_hypothesis
from .sanity import sanitize_newdata, sanitize_vcov, get_variables_names, sanitize_by
from .transform import get_transform
from .uncertainty import get_jacobian, get_se, get_z_p_ci
from .utils import sort_columns, get_modeldata, get_pad, upcast
from .classes import MarginaleffectsDataFrame


def get_predictions(model, params, newdata: pl.DataFrame):
    if isinstance(newdata, np.ndarray):
        exog = newdata
    else:
        y, exog = patsy.dmatrices(model.model.formula, newdata.to_pandas())
    p = model.model.predict(params, exog)
    if p.ndim == 1:
        p = pl.DataFrame({"rowid": range(newdata.shape[0]), "estimate": p})
    elif p.ndim == 2:
        colnames = {f"column_{i}": str(i) for i in range(p.shape[1])}
        p = (
            pl.DataFrame(p)
            .rename(colnames)
            .with_columns(pl.Series(range(p.shape[0]), dtype=pl.Int32).alias("rowid"))
            .melt(id_vars="rowid", variable_name="group", value_name="estimate")
        )
    else:
        raise ValueError(
            "The `predict()` method must return an array with 1 or 2 dimensions."
        )
    p = p.with_columns(pl.col("rowid").cast(pl.Int32))
    return p


def predictions(
    model,
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
    pass

    # sanity checks
    by = sanitize_by(by)
    V = sanitize_vcov(vcov, model)
    newdata = sanitize_newdata(model, newdata, wts=wts, by=by)

    # pad
    modeldata = get_modeldata(model)
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
    def fun(x):
        out = get_predictions(model, np.array(x), exog)
        out = get_by(model, out, newdata=newdata, by=by, wts=wts)
        out = get_hypothesis(out, hypothesis=hypothesis)
        return out

    out = fun(model.params)
    if vcov is not None:
        J = get_jacobian(fun, model.params)
        se = get_se(J, V)
        out = out.with_columns(pl.Series(se).alias("std_error"))
        out = get_z_p_ci(out, model, conf_level=conf_level)
    out = get_transform(out, transform=transform)
    out = get_equivalence(out, equivalence=equivalence)
    out = sort_columns(out, by=by)

    # unpad
    if "rowid" in out.columns and pad.shape[0] > 0:
        out = out[:-pad.shape[0]:]

    out = MarginaleffectsDataFrame(out, by=by, conf_level=conf_level)
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
