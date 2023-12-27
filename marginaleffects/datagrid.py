from functools import reduce, partial

import polars as pl

from .sanitize_model import sanitize_model


def datagrid(
    model=None,
    newdata=None,
    grid_type="mean_or_mode",
    FUN_numeric=lambda x: x.mean(),
    FUN_other=lambda x: x.mode()[0],  # mode can return multiple values
    **kwargs,
):
    """
    Data grids

    Generate a data grid of user-specified values for use in the 'newdata' argument of the 'predictions()', 'comparisons()', and 'slopes()' functions. This is useful to define where in the predictor space we want to evaluate the quantities of interest. Ex: the predicted outcome or slope for a 37 year old college graduate.

    Parameters:

    - `model`: Model object
    - `newdata`: DataFrame (one and only one of the `model` and `newdata` arguments can be used.)
    - `grid_type`: Determines the functions to apply to each variable. The defaults can be overridden by defining individual variables explicitly in `...`, or by supplying a function to one of the `FUN_*` arguments.
        * "mean_or_mode": Character, factor, logical, and binary variables are set to their modes. Numeric, integer, and other variables are set to their means.
        * "balanced": Each unique level of character, factor, logical, and binary variables are preserved. Numeric, integer, and other variables are set to their means. Warning: When there are many variables and many levels per variable, a balanced grid can be very large. In those cases, it is better to use `grid_type="mean_or_mode"` and to specify the unique levels of a subset of named variables explicitly.
        * "counterfactual": the entire dataset is duplicated for each combination of the variable values specified in `...`. Variables not explicitly supplied to `datagrid()` are set to their observed values in the original dataset.
    - `FUN_numeric`: The function to be applied to numeric variables.
    - `FUN_other`: The function to be applied to other variable types.

    Returns:

    A Polars DataFrame in which each row corresponds to one combination of the named predictors supplied by the user. Variables which are not explicitly defined are held at their mean or mode.

    Examples:

    ```python
    import polars as pl
    import statsmodels.formula.api as smf
    from marginaleffects import *
    mtcars = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv")

    # The output only has 2 rows, and all the variables except `hp` are at their mean or mode.
    datagrid(newdata = mtcars, hp = [100, 110])

    # We get the same result by feeding a model instead of a DataFrame
    mod = smf.ols("mpg ~ hp * qsec", mtcars).fit()
    datagrid(model = mod, hp = [100, 110])

    # Use in `marginaleffects` to compute "Typical Marginal Effects". When used in `slopes()` or `predictions()` we do not need to specify the `model` or `newdata` arguments.
    nd = datagrid(mod, hp = [100, 110])
    slopes(mod, newdata = nd)

    # The full dataset is duplicated with each observation given counterfactual values of 100 and 110 for the `hp` variable. The original `mtcars` includes 32 rows, so the resulting dataset includes 64 rows.
    dg = datagrid(newdata = mtcars, hp = [100, 110], grid_type = "counterfactual")
    print(dg.shape)
    """

    # allow preditions() to pass `model` argument automatically
    if model is None and newdata is None:
        out = partial(
            datagrid,
            grid_type=grid_type,
            FUN_numeric=FUN_numeric,
            FUN_other=FUN_other,
            **kwargs,
        )
        return out

    msg = "grid_type must be 'mean_or_mode', 'balanced', or 'counterfactual'"
    assert isinstance(grid_type, str) and grid_type in [
        "mean_or_mode",
        "explicit",
        "counterfactual",
    ], msg

    if model is None and newdata is None:
        raise ValueError("One of model or newdata must be specified")
    else:
        model = sanitize_model(model)

    if newdata is None:
        newdata = model.modeldata

    if grid_type == "counterfactual":
        return datagridcf(model=model, newdata=newdata, **kwargs)

    elif grid_type == "mean_or_mode":
        if FUN_numeric is None:

            def FUN_numeric(x):
                x.mean()

        if FUN_other is None:
            # mode can return multiple values
            def FUN_other(x):
                x.mode()[0]

    elif grid_type == "balanced":
        if FUN_numeric is None:

            def FUN_numeric(x):
                x.mean()

        if FUN_other is None:
            # mode can return multiple values
            def FUN_other(x):
                x.unique()[0]

    out = {}
    for key, value in kwargs.items():
        out[key] = pl.DataFrame({key: value})

    numtypes = [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    ]

    for col in newdata.columns:
        # not specified manually
        if col not in out.keys():
            # numeric
            if newdata[col].dtype in numtypes:
                out[col] = pl.DataFrame({col: FUN_numeric(newdata[col])})
            # other
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
        newdata = model.modeldata

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
