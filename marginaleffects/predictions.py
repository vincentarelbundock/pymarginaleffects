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
):
    """
    # `predictions()`

    `predictions()` and `avg_predictions()` predict outcomes using a fitted model on a specified scale for given combinations of values of predictor variables, such as their observed values, means, or factor levels (reference grid).

    * `predictions()`: unit-level (conditional) estimates.
    * `avg_predictions()`: average (marginal) estimates.

    See the package website and vignette for examples:

    - https://marginaleffects.com/chapters/predictions.html
    - https://marginaleffects.com

    ## Parameters

    `model`: (model object) Object fitted using the `statsmodels` formula API.

    `variables`: (str, list, dictionary) Specifies what variables (columns) to vary in order to make the prediction.

    - `None`: predictions are computed for all regressors in the model object (can be slow). Acceptable values depend on the variable type. See the examples below.
    - List[str] or str: List of variable names to compute predictions for.
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
        + `variables = "gear" : "pairwise", "hp" : 10`
        + `variables = "gear" : "sequential", "hp" : [100, 120]`

    `newdata`: (None, DataFrame, str) Data frame or string specifying where statistics are evaluated in the predictor space.

    - None: Compute predictions at each observed value in the original dataset (empirical distribution)
    - Dataframe: should be created with datagrid() function
    - str:
        * "mean": Compute predictions at the mean of the regressor
        * "median": Compute predictions at the median of the regressor
        * "balanced": Compute predictions on a balanced grid with every combination of categories and numeric variables held at their means.
        * "tukey": Probably NotImplemented
        * "grid": Probably NotImplemented

    `by`: (bool, List[str], optional) A logical value or a list of column names in `newdata`.

    - `True`: estimate is aggregated across the whole dataset.
    - list: estimates are aggregated for each unique combination of values in the columns.

    `transform`: (function) Function specifying a transformation applied to unit-level estimates and confidence intervals just before the function returns results. Functions must accept a full column (series) of a Polars data frame and return a corresponding series of the same length. Ex:

    - `transform = numpy.exp`
    - `transform = lambda x: x.exp()`
    - `transform = lambda x: x.map_elements()`

    `hypothesis`: (str, int, float, numpy array) Specifies a hypothesis test or custom contrast

    * Number to specify the null hypothesis.
    * Numpy array with a number of rows equal to the number of estimates.
    * String equation with an equal sign and estimate number in b0, b1, b2, etc. format.
        - "b0 = b1"
        - "b0 - (b1 + b2) = 0"
    * Two-side formula like "ratio ~ reference"
        - Left-hand side: "ratio", "difference"
        - Right-hand side: 'reference', 'sequential', 'pairwise', 'revreference', 'revsequential', 'revpairwise'

    - int, float: The null hypothesis used in the computation of Z and p-values (before applying transform)
    - str:
        * equation specifying linear or non-linear hypothesis tests. Use the names of the model variables, or use `b0`, `b1` to identify the position of each parameter. The `b*` wildcard can be used to test hypotheses on all estimates. Examples:
            - `hp = drat`
            - `hp + drat = 12`
            - `b0 + b1 + b2 = 0`
            - `b* / b0 = 1`
        * one of the following hypothesis test strings:
            - `pairwise` and `revpairwise`: pairwise differences between estimates in each row.
            - `reference` and `revreference`: differences between the estimates in each row and the estimate in the first row.
            - `sequential` and `revsequential`: differences between an estimate and the estimate in the next row.
    - numpy.ndarray: Each column is a vector of weights. The output is the dot product between these vectors of weights and the vectors of estimates. e.g. `hypothesis=np.array([[1, 1, 2], [2, 2, 3]]).T`
    - See the Examples section and the vignette: https://marginaleffects.com/chapters/hypothesis.html

    `wts`: (str, optional) Column name of weights to use for marginalization. Must be a column in `newdata`.

    `vcov`: (bool, np.ndarray, default=True) Type of uncertainty estimates to report (e.g. for robust standard errors). Acceptable values are:

    - `True`: Use the model's default covariance matrix.
    - `False`: Do not compute standard errors.
    - String: Literal indicating the kind of uncertainty estimates to return:
        - Heteroskedasticity-consistent: `"HC0"`, `"HC1"`, `"HC2"`, `"HC3"`.
    - np.ndarray: A custom square covariance matrix.

    `equivalence`: (list, optional) List of 2 numeric float values specifying the bounds used for the two-one-sided test (TOST) of equivalence, and for the non-inferiority and non-superiority tests. See the Details section below.

    `conf_level`: (float, default=0.95) Numeric value specifying the confidence level for the confidence intervals.

    `eps_vcov`: (float) optional custom value for the finite difference approximation of the jacobian matrix. By default, the function uses the square root of the machine epsilon.

    ## Returns

    A Polars DataFrame with (some of) the following columns:

    - `term`: the name of the variable.
    - `contrast`: the comparison method used.
    - `estimate`: the estimated contrast, difference, ratio, or other transformation between pairs of predictions.
    - `std_error`: the standard error of the estimate.
    - `statistic`: the test statistic (estimate / std.error).
    - `p_value`: the p-value of the test.
    - `s_value`: Shannon transform of the p value.
    - `conf_low`: the lower confidence interval bound.
    - `conf_high`: the upper confidence interval bound.
    - `pred_low`: the lower prediction interval bound.
    - `pred_high`: the upper prediction interval bound.


    ## Details

    ### Two-One-Sided Test (TOST) of Equivalence

    The `equivalence` argument specifies the bounds used for the two-one-sided test (TOST) of equivalence, and for the non-inferiority and non-superiority tests. The first element specifies the lower bound, and the second element specifies the upper bound. If `None`, equivalence tests are not performed.

    ### Order of operations.

    Behind the scenes, the arguments of `marginaleffects` functions are evaluated in this order:

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
    variables=None,
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
    # `predictions()`

    `predictions()` and `avg_predictions()` predict outcomes using a fitted model on a specified scale for given combinations of values of predictor variables, such as their observed values, means, or factor levels (reference grid).

    * `predictions()`: unit-level (conditional) estimates.
    * `avg_predictions()`: average (marginal) estimates.

    See the package website and vignette for examples:

    - https://marginaleffects.com/chapters/predictions.html
    - https://marginaleffects.com

    ## Parameters

    `model`: (model object) Object fitted using the `statsmodels` formula API.

    `variables`: (str, list, dictionary) Specifies what variables (columns) to vary in order to make the prediction.

    - `None`: predictions are computed for all regressors in the model object (can be slow). Acceptable values depend on the variable type. See the examples below.
    - List[str] or str: List of variable names to compute predictions for.
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
        + `variables = "gear" : "pairwise", "hp" : 10`
        + `variables = "gear" : "sequential", "hp" : [100, 120]`

    `newdata`: (None, DataFrame, str) Data frame or string specifying where statistics are evaluated in the predictor space.

    - None: Compute predictions at each observed value in the original dataset (empirical distribution)
    - Dataframe: should be created with datagrid() function
    - str:
        * "mean": Compute predictions at the mean of the regressor
        * "median": Compute predictions at the median of the regressor
        * "balanced": Compute predictions on a balanced grid with every combination of categories and numeric variables held at their means.
        * "tukey": Probably NotImplemented
        * "grid": Probably NotImplemented

    `by`: (bool, List[str], optional) A logical value or a list of column names in `newdata`.

    - `True`: estimate is aggregated across the whole dataset.
    - list: estimates are aggregated for each unique combination of values in the columns.

    `transform`: (function) Function specifying a transformation applied to unit-level estimates and confidence intervals just before the function returns results. Functions must accept a full column (series) of a Polars data frame and return a corresponding series of the same length. Ex:

    - `transform = numpy.exp`
    - `transform = lambda x: x.exp()`
    - `transform = lambda x: x.map_elements()`

    `hypothesis`: (str, int, float, numpy array) Specifies a hypothesis test or custom contrast

    * Number to specify the null hypothesis.
    * Numpy array with a number of rows equal to the number of estimates.
    * String equation with an equal sign and estimate number in b0, b1, b2, etc. format.
        - "b0 = b1"
        - "b0 - (b1 + b2) = 0"
    * Two-side formula like "ratio ~ reference"
        - Left-hand side: "ratio", "difference"
        - Right-hand side: 'reference', 'sequential', 'pairwise', 'revreference', 'revsequential', 'revpairwise'

    - int, float: The null hypothesis used in the computation of Z and p-values (before applying transform)
    - str:
        * equation specifying linear or non-linear hypothesis tests. Use the names of the model variables, or use `b0`, `b1` to identify the position of each parameter. The `b*` wildcard can be used to test hypotheses on all estimates. Examples:
            - `hp = drat`
            - `hp + drat = 12`
            - `b0 + b1 + b2 = 0`
            - `b* / b0 = 1`
        * one of the following hypothesis test strings:
            - `pairwise` and `revpairwise`: pairwise differences between estimates in each row.
            - `reference` and `revreference`: differences between the estimates in each row and the estimate in the first row.
            - `sequential` and `revsequential`: differences between an estimate and the estimate in the next row.
    - numpy.ndarray: Each column is a vector of weights. The output is the dot product between these vectors of weights and the vectors of estimates. e.g. `hypothesis=np.array([[1, 1, 2], [2, 2, 3]]).T`
    - See the Examples section and the vignette: https://marginaleffects.com/chapters/hypothesis.html

    `wts`: (str, optional) Column name of weights to use for marginalization. Must be a column in `newdata`.

    `vcov`: (bool, np.ndarray, default=True) Type of uncertainty estimates to report (e.g. for robust standard errors). Acceptable values are:

    - `True`: Use the model's default covariance matrix.
    - `False`: Do not compute standard errors.
    - String: Literal indicating the kind of uncertainty estimates to return:
        - Heteroskedasticity-consistent: `"HC0"`, `"HC1"`, `"HC2"`, `"HC3"`.
    - np.ndarray: A custom square covariance matrix.

    `equivalence`: (list, optional) List of 2 numeric float values specifying the bounds used for the two-one-sided test (TOST) of equivalence, and for the non-inferiority and non-superiority tests. See the Details section below.

    `conf_level`: (float, default=0.95) Numeric value specifying the confidence level for the confidence intervals.

    `eps_vcov`: (float) optional custom value for the finite difference approximation of the jacobian matrix. By default, the function uses the square root of the machine epsilon.

    ## Returns

    A Polars DataFrame with (some of) the following columns:

    - `term`: the name of the variable.
    - `contrast`: the comparison method used.
    - `estimate`: the estimated contrast, difference, ratio, or other transformation between pairs of predictions.
    - `std_error`: the standard error of the estimate.
    - `statistic`: the test statistic (estimate / std.error).
    - `p_value`: the p-value of the test.
    - `s_value`: Shannon transform of the p value.
    - `conf_low`: the lower confidence interval bound.
    - `conf_high`: the upper confidence interval bound.
    - `pred_low`: the lower prediction interval bound.
    - `pred_high`: the upper prediction interval bound.


    ## Details

    ### Two-One-Sided Test (TOST) of Equivalence

    The `equivalence` argument specifies the bounds used for the two-one-sided test (TOST) of equivalence, and for the non-inferiority and non-superiority tests. The first element specifies the lower bound, and the second element specifies the upper bound. If `None`, equivalence tests are not performed.

    ### Order of operations.

    Behind the scenes, the arguments of `marginaleffects` functions are evaluated in this order:

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
## Details
"""
    + DocsDetails.docstring_tost
    + DocsDetails.docstring_order_of_operations
)


predictions.__doc__ = docs_predictions

avg_predictions.__doc__ = predictions.__doc__
