from functools import reduce, partial

import polars as pl

from .sanitize_model import sanitize_model
import marginaleffects.utils as ut


def datagrid(
    model=None,
    newdata=None,
    grid_type="mean_or_mode",
    FUN_binary=None,
    FUN_character=None,
    FUN_numeric=None,
    FUN_other=None,
    **kwargs,
):
    """
    Generate a data grid of user-specified values for use in the 'newdata' argument of the 'predictions()', 'comparisons()', and 'slopes()' functions.

    For more information, visit the website: https://marginaleffects.com/

    Or type: `help(datagrid)`
    """

    # allow predictions() to pass `model` argument automatically
    if model is None and newdata is None:
        out = partial(
            datagrid,
            grid_type=grid_type,
            FUN_binary=FUN_binary,
            FUN_character=FUN_character,
            FUN_numeric=FUN_numeric,
            FUN_other=FUN_other,
            **kwargs,
        )
        return out

    msg = "grid_type must be 'mean_or_mode', 'balanced', or 'counterfactual'"
    assert isinstance(grid_type, str) and grid_type in [
        "mean_or_mode",
        "balanced",
        "counterfactual",
    ], msg

    if model is None and newdata is None:
        raise ValueError("One of model or newdata must be specified")
    else:
        model = sanitize_model(model)

    if newdata is None:
        newdata = model.data

    if grid_type == "counterfactual":
        return datagridcf(model=model, newdata=newdata, **kwargs)

    elif grid_type == "mean_or_mode":
        pass

    elif grid_type == "balanced":
        if FUN_binary is None:

            def FUN_binary(x):
                return x.unique()

        if FUN_character is None:

            def FUN_character(x):
                return x.unique()

        if FUN_numeric is None:

            def FUN_numeric(x):
                return x.mean()

        if FUN_other is None:

            def FUN_other(x):
                return x.unique()

    if FUN_binary is None:

        def FUN_binary(x):
            return x.mode()[0]

    if FUN_character is None:

        def FUN_character(x):
            return x.mode()[0]

    if FUN_numeric is None:

        def FUN_numeric(x):
            return x.mean()

    if FUN_other is None:

        def FUN_other(x):
            return x.mode()[0]

    out = {}
    for key, value in kwargs.items():
        if value is not None:
            out[key] = pl.DataFrame({key: value})

    # Balanced grid should not be built with combiations of response variable, otherwise we get a
    # duplicated rows on `grid_type="balanced"` and other types.
    if grid_type == "balanced":
        if hasattr(model, "response_name") and isinstance(model.response_name, str):
            col = model.response_name
            out[col] = pl.DataFrame({col: newdata[col].mode()[0]})

    if model is not None:
        variables_type = model.variables_type
    else:
        variables_type = ut.get_type_dictionary(formula=None, modeldata=newdata)

    cols = newdata.columns
    cols = [col for col in cols if col in variables_type.keys()]
    cols = [col for col in cols if col in newdata.columns]
    cols = [col for col in cols if col not in out.keys()]

    for col in cols:
        if variables_type[col] == "binary":
            out[col] = pl.DataFrame({col: FUN_binary(newdata[col])})
        elif variables_type[col] in ["integer", "numeric"]:
            out[col] = pl.DataFrame({col: FUN_numeric(newdata[col])})
        elif variables_type[col] == "character":
            out[col] = pl.DataFrame({col: FUN_character(newdata[col])})
        else:
            out[col] = pl.DataFrame({col: FUN_other(newdata[col])})

    out = reduce(lambda x, y: x.join(y, how="cross"), out.values())

    out.datagrid_explicit = list(kwargs.keys())

    return out


def datagridcf(model=None, newdata=None, **kwargs):
    if model is None and newdata is None:
        raise ValueError("One of model or newdata must be specified")

    model = sanitize_model(model)

    if newdata is None:
        newdata = model.data

    if "rowid" not in newdata.columns:
        newdata = newdata.with_columns(
            pl.Series(range(newdata.shape[0])).alias("rowid")
        )
    newdata = newdata.rename({"rowid": "rowidcf"})

    # Create dataframe from kwargs
    dfs = [pl.DataFrame({k: v}) for k, v in kwargs.items()]

    # Perform cross join
    df_cross = reduce(lambda df1, df2: df1.join(df2, how="cross"), dfs)

    # Drop would-be duplicates
    newdata = newdata.drop(df_cross.columns)

    result = newdata.join(df_cross, how="cross")

    result.datagrid_explicit = list(kwargs.keys())

    return result


datagrid.__doc__ = """
    # `datagrid()`

    Generate a data grid of user-specified values for use in the 'newdata' argument of the 'predictions()', 'comparisons()', and 'slopes()' functions.

    This is useful to define where in the predictor space we want to evaluate the quantities of interest. Ex: the predicted outcome or slope for a 37 year old college graduate.

    ## Parameters
    * model: (object, optional)
        Model object.
        * (one and only one of the `model` and `newdata` arguments can be used.)
    * newdata: (DataFrame, optional)
        Data frame used to define the predictor space.
        * (one and only one of the `model` and `newdata` arguments can be used.)
    * grid_type: (str, optional)
        Determines the functions to apply to each variable. The defaults can be overridden by defining individual variables explicitly in the `**kwargs`, or by supplying a function to one of the `FUN_*` arguments.
        * "mean_or_mode": Character, factor, logical, and binary variables are set to their modes. Numeric, integer, and other variables are set to their means.
        * "balanced": Each unique level of character, factor, logical, and binary variables are preserved. Numeric, integer, and other variables are set to their means. Warning: When there are many variables and many levels per variable, a balanced grid can be very large. In those cases, it is better to use `grid_type="mean_or_mode"` and to specify the unique levels of a subset of named variables explicitly.
        * "counterfactual": the entire dataset is duplicated for each combination of the variable values specified in `**kwargs`. Variables not explicitly supplied to `datagrid()` are set to their observed values in the original dataset.
    * FUN_numeric: (Callable, optional)
        The function to be applied to numeric variables.
    * FUN_other: (Callable, optional)
        The function to be applied to other variable types.
    * **kwargs
        * Named arguments where the name is the variable name and the value is a list of values to use in the grid. If a variable is not specified, it is set to its mean or mode depending on the `grid_type` argument.

    ## Returns
    (polars.DataFrame)
    * DataFrame where each row corresponds to one combination of the named predictors supplied by the user. Variables which are not explicitly defined are held at their mean or mode.

    ## Examples
    ```py
    import polars as pl
    import statsmodels.formula.api as smf
    from marginaleffects import *
    data = get_dataset("thornton")

    # The output only has 2 rows, and all the variables except `hp` are at their mean or mode.
    datagrid(newdata = data, village = [43, 11])

    # We get the same result by feeding a model instead of a DataFrame
    mod = smf.ols("outcome ~ incentive + distance", data).fit()
    datagrid(model = mod, village = [43, 11])

    # Use in `marginaleffects` to compute "Typical Marginal Effects". When used in `slopes()` or `predictions()` we do not need to specify the `model` or `newdata` arguments.
    nd = datagrid(mod, village = [43, 11])
    slopes(mod, newdata = nd)

    # The full dataset is duplicated with each observation given counterfactual values of 43 and 11 for the `village` variable. 
    # The original `thornton` includes 2884 rows, so the resulting dataset includes 5768 rows.
    dg = datagrid(newdata = data, village = [43, 11], grid_type = "counterfactual")
    dg.shape
    ```
    """
