import numpy as np
import polars as pl

from .by import get_by
from .classes import MarginaleffectsDataFrame
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
from .utils import sort_columns
from .model_pyfixest import ModelPyfixest
from .model_linearmodels import ModelLinearmodels
from .formulaic_utils import model_matrices


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
):
    """
    Predict outcomes using a fitted model on a specified scale for given combinations of values
    of predictor variables, such as their observed values, means, or factor levels (reference grid).

    This function handles unit-level (conditional) estimates based
    on the `variables` and `newdata` arguments. See the package website and vignette for examples:
        - https://marginaleffects.com/chapters/predictions.html
        - https://marginaleffects.com

    Parameters
    ----------
    model : object
        Model object.
    variables : dict, optional
        Dictionary of variables and associated values over which to compute predictions.
        If `None`, computes one prediction per row in `newdata`.
        Note that the `values` accept the following special strings:
            - "sd": Contrast across one standard deviation around the regressor mean.
            - "2sd": Contrast across two standard deviations around the regressor mean.
            - "iqr": Contrast across the interquartile range of the regressor.
            - "minmax": Contrast between the maximum and the minimum values of the regressor.
            - "threenum": mean and 1 standard deviation on both sides
            - "fivenum": Tukey's five numbers
    newdata : Union[None, DataFrame], optional
        Grid of predictor values at which to evaluate predictions, by default predictions are made on the data used to fit the model.
        - Dataframe: should be created with datagrid() function
        - String:
            * "mean": Compute predictions at the mean of the regressor
            * "median": Compute predictions at the median of the regressor
            * "balanced": Predictions evaluated on a balanced grid with every combination of categories and numeric variables held at their means.
            * "tukey": Probably NotImplemented
            * "grid": Probably NotImplemented
    by : bool, List[str], optional
        a logical value or a list of column names in `newdata`. If `True`, estimate is aggregated across the whole dataset. If a list is provided, estimates are aggregated for each unique combination of values in the columns.
    transform : Callable, optional
        a function specifying a transformation applied to unit-level estimates and confidence intervals just before the function returns results. Functions must accept a full column (series) of a Polars data frame and return a corresponding series of the same length. Ex:
            - `transform = numpy.exp`
            - `transform = lambda x: x.exp()`
            - `transform = lambda x: x.map_elements()`
    hypothesis : str, optional
        String formula of hypothesis test or numpy array.
    wts : str, optional
        Column name of weights to use for marginalization. Must be a column in `newdata`.
    vcov : bool, np.ndarray, optional
        Type of uncertainty estimates to report (e.g. for robust standard errors). Acceptable values are:
            - `True`: Use the model's default covariance matrix.
            - `False`: Do not compute standard errors.
            - np.ndarray: A custom square covariance matrix.
    equivalence : list, optional
        List of 2 numeric values specifying the bounds used for the two-one-sided test (TOST) of equivalence, and for the non-inferiority and non-superiority tests. See the Notes section below.
    eps_vcov : float, optional
        Custom value for the finite difference approximation of the Jacobian matrix. By default, the function uses the square root of the machine epsilon.

    Notes
    -----
    - The `equivalence` argument specifies the bounds used for the two-one-sided test (TOST) of equivalence, and for the non-inferiority and non-superiority tests. The first element specifies the lower bound, and the second element specifies the upper bound. If `None`, equivalence tests are not performed.

    - Order of operations. Behind the scenes, the arguments of `marginaleffects` functions are evaluated in this order:
        1. `newdata`
        2. `variables`
        3. `comparison` and `slope`
        4. `by`
        5. `vcov`
        6. `hypothesis`
        7. `transform`

    """

    if callable(newdata):
        newdata = newdata(model)

    # sanity checks
    model = sanitize_model(model)
    by = sanitize_by(by)
    V = sanitize_vcov(vcov, model)
    newdata = sanitize_newdata(model, newdata, wts=wts, by=by)
    hypothesis_null = sanitize_hypothesis_null(hypothesis)

    modeldata = model.data

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
            f = model.formula
        endog, exog = model_matrices(f, newdata, formula_engine=model.formula_engine)

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

    out = MarginaleffectsDataFrame(
        out, by=by, conf_level=conf_level, jacobian=J, newdata=newdata
    )
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
    """
    Predict outcomes using a fitted model on a specified scale for given combinations of values
    of predictor variables, such as their observed values, means, or factor levels (reference grid).

    This function handles average (marginal) estimates based
    on the `variables` and `newdata` arguments. See the package website and vignette for examples:
        - https://marginaleffects.com/chapters/predictions.html
        - https://marginaleffects.com

    """

    if callable(newdata):
        newdata = newdata(model)

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


# inherit_numpy_docstring(_template_returns.__doc__, predictions)
# inherit_numpy_docstring(predictions.__doc__, avg_predictions)
