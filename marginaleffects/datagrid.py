from functools import reduce
from .utils import get_modeldata

import polars as pl


def datagrid(
    model=None,
    newdata=None,
    FUN_numeric=lambda x: x.mean(),
    FUN_other=lambda x: x.mode()[0],  # mode can return multiple values
    **kwargs
):
    """
    Data grids

    Generate a data grid of user-specified values for use in the 'newdata' argument of the 'predictions()', 'comparisons()', and 'slopes()' functions. This is useful to define where in the predictor space we want to evaluate the quantities of interest. Ex: the predicted outcome or slope for a 37 year old college graduate.

    - `datagrid()` generates data frames with combinations of "typical" or user-supplied predictor values.
    - `datagridcf()` generates "counter-factual" data frames, by replicating the entire dataset once for every combination of predictor values supplied by the user.

    Parameters:

    - `model`: Model object
    - `newdata`: DataFrame (one and only one of the `model` and `newdata` arguments can be used.)
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
    dg = datagridcf(newdata = mtcars, hp = [100, 110])
    print(dg.shape)
    """

    if model is None and newdata is None:
        raise ValueError("One of model or newdata must be specified")

    if newdata is None:
        newdata = get_modeldata(model)

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
            if newdata[col].dtype() in numtypes:
                out[col] = pl.DataFrame({col: FUN_numeric(newdata[col])})
            # other
            else:
                out[col] = pl.DataFrame({col: FUN_other(newdata[col])})

    out = reduce(lambda x, y: x.join(y, how="cross"), out.values())

    return out



def datagridcf(model=None, newdata=None, **kwargs):
    """
    Data grids

    Generate a data grid of user-specified values for use in the 'newdata' argument of the 'predictions()', 'comparisons()', and 'slopes()' functions. This is useful to define where in the predictor space we want to evaluate the quantities of interest. Ex: the predicted outcome or slope for a 37 year old college graduate.

    - `datagrid()` generates data frames with combinations of "typical" or user-supplied predictor values.
    - `datagridcf()` generates "counter-factual" data frames, by replicating the entire dataset once for every combination of predictor values supplied by the user.

    Parameters:

    - `model`: Model object
    - `newdata`: DataFrame (one and only one of the `model` and `newdata` arguments can be used.)

    Returns:

    A Polars DataFrame in which each row corresponds to one combination of the named predictors supplied by the user. Variables which are not explicitly defined are held at their mean or mode.
    """

    if model is None and newdata is None:
        raise ValueError("One of model or newdata must be specified")

    if newdata is None:
        newdata = get_modeldata(model)

    if "rowid" not in newdata.columns:
        newdata = newdata.with_columns(pl.Series(range(newdata.shape[0])).alias("rowid"))

    # Create dataframe from kwargs
    dfs = [pl.DataFrame({k: v}) for k, v in kwargs.items()]

    # Perform cross join
    df_cross = reduce(lambda df1, df2: df1.join(df2, how='cross'), dfs)

    result = newdata.join(df_cross, how = "cross")

    # Create rowid and rowidcf
    result = result.with_columns(pl.Series(range(result.shape[0])).alias("rowidcf"))

    return result