import polars as pl
import numpy as np
import sys

from .utils import get_modeldata, get_variable_type
from .datagrid import datagrid

import statsmodels.formula.api as smf




def build_plot(model, condition):

    modeldata = get_modeldata(model)

    assert 1 <= len(condition) <= 3, f"Lenght of condition must be inclusively between 1 and 3. Got : {len(condition)}"

    to_datagrid = {}


    if isinstance(condition, list):

        assert all(ele in modeldata.columns for ele in condition), "All elements of condition must be columns of the model"

        for i, ele in enumerate(condition):
            variable_type = get_variable_type(ele, modeldata)

            if variable_type == 'numeric':
                if i == 0:
                    to_datagrid[ele] = np.linspace(modeldata[ele].min(), modeldata[ele].max(), 100).tolist()
                else:
                    to_datagrid[ele] = np.percentile(modeldata[ele], [0, 25, 50, 75, 100], method="midpoint").tolist()

            elif variable_type == 'boolean' or variable_type == 'character':
                to_datagrid[ele] = modeldata[ele].unique().to_list()
                assert len(to_datagrid[ele]) <= 10, f"Character type variables of more than 10 unique values are not supported. {ele} variable has {len(to_datagrid[ele])} unique values."

            # binary ?


    elif isinstance(condition, dict):
        assert all(key in modeldata.columns for key in condition.keys()), "All keys of condition must be columns of the model"

        to_datagrid = condition    


    # uncomfortable. did you already solve this problem?
    # also not sure if I use datagrid the right way
    dt_code = "datagrid(newdata=modeldata"
    for key, value in to_datagrid.items():
        dt_code += ", " + key + "=" + str(value)
    dt_code += ")"

    exec("global dt; dt = " + dt_code)
    
    return dt
