import polars as pl
import numpy as np
import sys

from .utils import get_modeldata, get_variable_type
from .datagrid import datagrid


def build_plot(model, condition):


    modeldata = get_modeldata(model)

    if isinstance(condition, str):
        condition = [condition]

    assert 1 <= len(condition) <= 3, f"Lenght of condition must be inclusively between 1 and 3. Got : {len(condition)}"


    to_datagrid = {}
    first_key = ""  # special case when the first element is numeric

    if isinstance(condition, list):
        assert all(ele in modeldata.columns for ele in condition), "All elements of condition must be columns of the model"
        first_key = condition[0]
        to_datagrid = {key: None for key in condition}

    elif isinstance(condition, dict):
        assert all(key in modeldata.columns for key in condition.keys()), "All keys of condition must be columns of the model"
        first_key = next(iter(condition))
        to_datagrid = condition    



    for key, value in to_datagrid.items():

        variable_type = get_variable_type(key, modeldata)

        # Check type of user-supplied dict values
        if value is not None:
            test_df = pl.DataFrame({key: value})
            assert variable_type == get_variable_type(key, test_df), f"Supplied data type of {key} column ({get_variable_type(key, test_df)}) does not match the type of the variable ({variable_type})."
            continue

        if variable_type == 'numeric':
            if key == first_key:
                to_datagrid[key] = np.linspace(modeldata[key].min(), modeldata[key].max(), 100).tolist()
            else:
                to_datagrid[key] = np.percentile(modeldata[key], [0, 25, 50, 75, 100], method="midpoint").tolist()

        elif variable_type == 'boolean' or variable_type == 'character':
            to_datagrid[key] = modeldata[key].unique().to_list()
            assert len(to_datagrid[key]) <= 10, f"Character type variables of more than 10 unique values are not supported. {ele} variable has {len(to_datagrid[ele])} unique values."



    dt_code = "datagrid(newdata=modeldata"
    for key, value in to_datagrid.items():
        dt_code += ", " + key + "="
        if isinstance(value, str):
            dt_code += "'" + value + "'"
        else:
            dt_code += str(value)
    dt_code += ")"

    exec("global dt; dt = " + dt_code)

    return dt
