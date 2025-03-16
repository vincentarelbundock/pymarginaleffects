from .docs import DocsParameters
from .plot_common import dt_on_condition, plot_labels
from .p9 import plot_common
from .predictions import predictions
from .sanitize_model import sanitize_model
import copy


def plot_predictions(
    model,
    condition=None,
    by=False,
    newdata=None,
    vcov=True,
    conf_level=0.95,
    transform=None,
    draw=True,
    wts=None,
):
    """
    # `plot_predictions()`

    Plot predictions on the y-axis against values of one or more predictors (x-axis, colors/shapes, and facets).

    The `by` argument is used to plot marginal predictions, that is, predictions made on the original data, but averaged
    by subgroups. This is analogous to using the `by` argument in the `predictions()` function.

    The `condition` argument is used to plot conditional predictions, that is, predictions made on a user-specified grid.
    This is analogous to using the `newdata` argument and `datagrid()` function in a `predictions()` call.

    All unspecified variables are held at their mean or mode. This includes grouping variables in mixed-effects models, so analysts who fit such models may want to specify the groups of interest using the `variables` argument, or supply model-specific arguments to compute population-level estimates. See details below.

    See the "Plots" vignette and website for tutorials and information on how to customize plots:

    - https://marginaleffects.com/articles/plot.html
    - https://marginaleffects.com


    ## Parameters

    `model`: (model object) Object fitted using the `statsmodels` formula API.

    `condition`: (str, list, dictionary) Max length : 4.

    - Position's representation:
        1. x-axis.
        2. color.
        3. facet (wrap if no fourth variable, otherwise cols of grid).
        4. facet (rows of grid).
    - Argument types:
        - list : Names of the predictors to display
            - Numeric variables in position 1 is summarized by 100 numbers
            - Numeric variables in positions 2, 3 and 4 are summarized by Tukey’s five numbers
        - dictionary : Keys correspond to predictors. Values can be one of the two below depending on predictor's type:
            - Series or list of the same type as the original variable.
            - Numeric variables:
                - String: "minmax", "threenum", "fivenum".
        - string : Same as list of length 1.

    `by`: (bool, str, list) Names of the categorical predictors to marginalize across. Max length of list is 4, with position meanings:

    1. x-axis.
    2. color.
    3. facet (wrap if no fourth variable, otherwise columns of grid).
    4. facet (rows of grid)

    `draw`: True returns a matplotlib plot. False returns a dataframe of the underlying data.

    `newdata`: (dataframe) When newdata is `None`, the grid is determined by the condition argument. When newdata is not `None`, the argument behaves in the same way as in the predictions() function.

    `wts`: (str, optional) Column name of weights to use for marginalization. Must be a column in `newdata`.

    `transform`: (function) Function specifying a transformation applied to unit-level estimates and confidence intervals just before the function returns results. Functions must accept a full column (series) of a Polars data frame and return a corresponding series of the same length. Ex:

    - `transform = numpy.exp`
    - `transform = lambda x: x.exp()`
    - `transform = lambda x: x.map_elements()`
    """
    model = sanitize_model(model)

    assert not (not by and newdata is not None), (
        "The `newdata` argument requires a `by` argument."
    )

    assert not (wts is not None and not by), (
        "The `wts` argument requires a `by` argument."
    )

    assert not (condition is None and not by), (
        "One of the `condition` and `by` arguments must be supplied, but not both."
    )

    # before dt_on_condition, which modifies in-place
    condition_input = copy.deepcopy(condition)

    if condition is not None:
        newdata = dt_on_condition(model, condition)

    dt = predictions(
        model,
        by=by,
        newdata=newdata,
        conf_level=conf_level,
        vcov=vcov,
        transform=transform,
        wts=wts,
    )

    dt = plot_labels(model, dt, condition_input)

    if not draw:
        return dt

    if isinstance(condition, str):
        var_list = [condition]
    elif isinstance(condition, list):
        var_list = condition
    elif isinstance(condition, dict):
        var_list = list(condition.keys())
    elif isinstance(by, str):
        var_list = [by]
    elif isinstance(by, list):
        var_list = by
    elif isinstance(by, dict):
        var_list = list(by.keys())

    # not sure why these get appended
    var_list = [x for x in var_list if x not in ["newdata", "model"]]

    assert len(var_list) < 5, (
        "The `condition` and `by` arguments can have a max length of 4."
    )

    return plot_common(model, dt, model.response_name, var_list=var_list)


plot_predictions.__doc__ = (
    """
# `plot_predictions()`
"""
    + DocsParameters.docstring_plot_intro("predictions")
    + """
## Parameters
"""
    + DocsParameters.docstring_model
    + DocsParameters.docstring_condition
    + DocsParameters.docstring_by_plot
    + DocsParameters.docstring_draw
    + DocsParameters.docstring_newdata_plot("predictions")
    + DocsParameters.docstring_wts
    + DocsParameters.docstring_transform
)
