from .comparisons import comparisons
from .plot_common import dt_on_condition
from .p9 import plot_common
from .sanitize_model import sanitize_model
from .plot_common import plot_labels
import copy


def plot_comparisons(
    model,
    condition=None,
    variables=None,
    newdata=None,
    comparison="difference",
    vcov=True,
    conf_level=0.95,
    by=False,
    wts=None,
    draw=True,
    hypothesis=None,
    equivalence=None,
    transform=None,
    eps=1e-4,
):
    """
    Plot comparisons on the y-axis against values of one or more predictors (x-axis, colors/shapes, and facets).

    The `by` argument is used to plot marginal comparisons, that is, comparisons made on the original data, but
    averaged by subgroups. This is analogous to using the `by` argument in the `comparisons()` function.

    The `condition` argument is used to plot conditional comparisons, that is, comparisons made on a
    user-specified grid. This is analogous to using the `newdata` argument and `datagrid()` function in a
    `comparisons()` call.

    All unspecified variables are held at their mean or mode. This includes grouping variables in mixed-effects
    models, so analysts who fit such models may want to specify the groups of interest using the variables argument,
    or supply model-specific arguments to compute population-level estimates. See details below.

    See the "Plots" vignette and website for tutorials and information on how to customize plots:
    - https://marginaleffects.com/articles/plot.html
    - https://marginaleffects.com

    Parameters
    ----------
    model : object
        Model object.

    variables : str, list, dictionary
        Name of the variable whose contrast we want to plot on the y-axis.
        Refer to the `comparisons()` documentation.

    condition : str, list, dictionary
        Max length : 3.
        1: x-axis. 2: color. 3: facets.
        list : Names of the predictors to display
            Numeric variables in position 1 is summarized by 100 numbers
            Numeric variables in positions 2 and 3 are summarized by Tukey’s five numbers
        dictionary : Keys correspond to predictors. Values are numeric vectors.
            - Series of lists of the same type as the original variable.
            - Numeric variables:
                - Series or list of numeric values.
                - String: "minmax", "threenum", "fivenum".

    by : bool, str, list
        Max length : 3.
        1: x-axis. 2: color. 3: facets.
        Aggregate unit-level estimates (aka, marginalize, average over).

    newdata : dataframe
        When newdata is NULL, the grid is determined by the condition argument. When newdata is not NULL, the argument behaves in the same way as in the predictions() function.

    wts: Column name of weights to use for marginalization. Must be a column in `newdata`

    transform : Callable, optional
        A function applied to unit-level adjusted predictions and confidence intervals just before
        the function returns results, by default None.

    draw : True returns a matplotlib plot. False returns a dataframe of the underlying data.
    """

    model = sanitize_model(model)

    assert not (
        not by and newdata is not None
    ), "The `newdata` argument requires a `by` argument."

    assert (condition is None and by) or (
        condition is not None and not by
    ), "One of the `condition` and `by` arguments must be supplied, but not both."

    assert not (
        wts is not None and not by
    ), "The `wts` argument requires a `by` argument."

    # before dt_on_condition, which modifies in-place
    condition_input = copy.deepcopy(condition)

    if condition is not None:
        newdata = dt_on_condition(model, condition)

    dt = comparisons(
        model,
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

    dt = plot_labels(dt, condition_input)

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

    assert (
        len(var_list) < 4
    ), "The `condition` and `by` arguments can have a max length of 3."

    if "contrast" in dt.columns and dt["contrast"].unique().len() > 1:
        var_list = var_list + ["contrast"]

    if not draw:
        return dt

    return plot_common(dt, "Comparison", var_list)
