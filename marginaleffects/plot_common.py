import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from .datagrid import datagrid  # noqa
from .sanitize_model import sanitize_model

from .utils import get_variable_type


def dt_on_condition(model, condition):
    model = sanitize_model(model)

    # not sure why newdata gets added
    modeldata = model.modeldata

    if isinstance(condition, str):
        condition = [condition]

    to_datagrid = {}
    first_key = ""  # special case when the first element is numeric

    if isinstance(condition, list):
        assert all(
            ele in modeldata.columns for ele in condition
        ), "All elements of condition must be columns of the model."
        first_key = condition[0]
        to_datagrid = {key: None for key in condition}

    elif isinstance(condition, dict):
        assert all(
            key in modeldata.columns for key in condition.keys()
        ), "All keys of condition must be columns of the model."
        first_key = next(iter(condition))
        to_datagrid = condition

    # not sure why `newdata` sometimes gets added
    if isinstance(condition, dict) and "newdata" in to_datagrid.keys():
        condition.pop("newdata", None)

    assert (
        1 <= len(condition) <= 3
    ), f"Lenght of condition must be inclusively between 1 and 3. Got : {len(condition)}."

    for key, value in to_datagrid.items():
        variable_type = get_variable_type(key, modeldata)

        # TODO: This too demanding for numeric with "threenum"
        # # Check type of user-supplied dict values
        # if value is not None:
        #     test_df = pl.DataFrame({key: value})
        #     assert (
        #         variable_type == get_variable_type(key, test_df)
        #     ), f"Supplied data type of {key} column ({get_variable_type(key, test_df)}) does not match the type of the variable ({variable_type})."
        #     continue

        if variable_type == "numeric":
            to_datagrid[key] = condition_numeric(
                modeldata, key, value, key == first_key
            )

        elif variable_type in ["boolean", "character", "binary"]:
            to_datagrid[key] = modeldata[key].unique().sort().to_list()
            assert (
                len(to_datagrid[key]) <= 10
            ), f"Character type variables of more than 10 unique values are not supported. {key} variable has {len(to_datagrid[key])} unique values."

    to_datagrid["newdata"] = modeldata
    dt = datagrid(**to_datagrid)
    return dt  # noqa: F821


def condition_numeric(modeldata, key, value, first):
    if value is None:
        if first:
            out = np.linspace(
                modeldata[key].min(), modeldata[key].max(), 100
            ).tolist()
        else:
            out = np.percentile(
                modeldata[key], [0, 25, 50, 75, 100], method="midpoint"
            ).tolist()
    elif isinstance(value, str) and value == "threenum":
        m = modeldata[key].mean()
        s = modeldata[key].std()
        out = [m - s, m, m + s]
    elif isinstance(value, str) and value == "fivenum":
        out = [0, .25, .5, .75, 1]
        out = [modeldata[key].quantile(x) for x in out]
    elif isinstance(value, str) and value == "minmax":
        out = [0, 1]
        out = [modeldata[key].quantile(x) for x in out]
    else:
        out = value

    return out
