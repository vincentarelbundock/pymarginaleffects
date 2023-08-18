import polars as pl
import numpy as np
import sys
import matplotlib.pyplot as plt

from .utils import get_modeldata, get_variable_type, find_response
from .datagrid import datagrid
from .predictions import predictions
from .slopes import slopes
from .comparisons import comparisons

import pytest

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

    return to_datagrid, dt


def filter_and_plot(dt, x_name, x_type, filter_expr=True, fig=None, axes_i=None, label=None):

    x = dt.filter(filter_expr).select(x_name).to_numpy().flatten()
    y = dt.filter(filter_expr).select("estimate").to_numpy().flatten()
    y_std = dt.filter(filter_expr).select("std_error").to_numpy().flatten()

    if fig is not None:
        plot_obj = fig.axes[axes_i] if axes_i is not None else plt
    else:
        fig = plt.figure()
        plot_obj = plt
    
    if x_type == "numeric":
        plot_obj.fill_between(x, y-y_std, y+y_std, alpha=0.2)
        plot_obj.plot(x, y, label=label)

    elif x_type == "character" or x_type == "boolean":
        plot_obj.errorbar(x, y, yerr=y_std, fmt='o', label=label)

    return fig


def plot_predictions(model, condition):

    titles_fontsize = 16

    grid, dt = build_plot(model, condition)

    con_names = list(grid.keys())

    for key in con_names:
        grid[key] = [grid[key]] if not isinstance(grid[key], list) else grid[key]

    pred_dt = predictions(model, newdata=dt)

    x_type = get_variable_type(con_names[0], pred_dt)

    if len(grid) == 3:
        fig, axes = plt.subplots(1, len(grid[con_names[2]]))

        for axes_i, con_2val in enumerate(grid[con_names[2]]):

            for con_1val in grid[con_names[1]]:
                filter_expr = (pl.col(con_names[2]) == con_2val) & (pl.col(con_names[1]) == con_1val)
                filter_and_plot(pred_dt, con_names[0], x_type, filter_expr=filter_expr, fig=fig, axes_i=axes_i, label=con_1val)

            fig.axes[axes_i].set_title(con_2val, fontsize=titles_fontsize)

        fig.axes[axes_i].legend(loc='center left', bbox_to_anchor=(1, 0.5), title=con_names[1], fontsize=titles_fontsize, title_fontsize=titles_fontsize)
                    
    elif len(grid) == 2:
        fig = plt.figure()
        ax = plt.subplot(111)

        for con_1val in grid[con_names[1]]:
            filter_expr = pl.col(con_names[1]) == con_1val
            filter_and_plot(pred_dt, con_names[0], x_type, filter_expr=filter_expr, fig=fig, label=con_1val)

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=con_names[1], fontsize=titles_fontsize, title_fontsize=titles_fontsize)

    elif len(grid) == 1:
        fig = filter_and_plot(pred_dt, con_names[0], x_type)

    else:
        raise ValueError("Condition's length must be inbetween 1 and 3.")

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel(con_names[0], fontsize=titles_fontsize)
    plt.ylabel(find_response(model), fontsize=titles_fontsize)

    plt.show()
