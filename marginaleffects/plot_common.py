import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .utils import get_modeldata, get_variable_type
from .datagrid import datagrid


def dt_on_condition(model, condition):


    modeldata = get_modeldata(model)

    if isinstance(condition, str):
        condition = [condition]

    assert 1 <= len(condition) <= 3, f"Lenght of condition must be inclusively between 1 and 3. Got : {len(condition)}."


    to_datagrid = {}
    first_key = ""  # special case when the first element is numeric

    if isinstance(condition, list):
        assert all(ele in modeldata.columns for ele in condition), "All elements of condition must be columns of the model."
        first_key = condition[0]
        to_datagrid = {key: None for key in condition}

    elif isinstance(condition, dict):
        assert all(key in modeldata.columns for key in condition.keys()), "All keys of condition must be columns of the model."
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


def plotter(dt, x_name, x_type, fig=None, axes_i=None, label=None, color=None):

    x = dt.select(x_name).to_numpy().flatten()
    y = dt.select("estimate").to_numpy().flatten()
    y_std = dt.select("std_error").to_numpy().flatten()

    if fig is not None:
        plot_obj = fig.axes[axes_i] if axes_i is not None else plt
    else:
        fig = plt.figure()
        plot_obj = plt
    
    if x_type == "numeric":
        if color is None:
            plot_obj.fill_between(x, y-y_std, y+y_std, alpha=0.2)
            plot_obj.plot(x, y, label=label)
        else:
            plot_obj.fill_between(x, y-y_std, y+y_std, color=color, alpha=0.2)
            plot_obj.plot(x, y, color=color, label=label)

    elif x_type == "character" or x_type == "boolean":
        if color is None:
            plot_obj.errorbar(x, y, yerr=y_std, fmt='o', label=label)
        else:
            plot_obj.errorbar(x, y, yerr=y_std, fmt='o', color=color, label=label)

    return fig


def plot_common(dt, y_label, x_name, color=None, subplot=None):

    titles_fontsize = 16

    x_type = get_variable_type(x_name, dt)

    if subplot is not None:
        fig, axes = plt.subplots(1, dt.n_unique(subset=[subplot]))
        color_i = 0
        color_dict = {}
        for axes_i, subplot_val in enumerate(dt.select(subplot).unique().to_numpy().flatten()):
            subplot_dt = dt.filter(pl.col(subplot) == subplot_val)

            if color is None:
                plotter(subplot_dt, x_name, x_type, fig=fig, axes_i=axes_i)

            else:
                for color_val in subplot_dt.select(color).unique().to_numpy().flatten():
                    if color_val not in color_dict:
                        color_dict[color_val] = plt.rcParams['axes.prop_cycle'].by_key()['color'][color_i]
                        color_i += 1

                    color_dt = subplot_dt.filter(pl.col(color)==color_val)

                    plotter(color_dt, x_name, x_type, fig=fig, axes_i=axes_i, label=color_val, color=color_dict[color_val])

            fig.axes[axes_i].set_title(subplot_val, fontsize=titles_fontsize)
        if color is not None:
            legend_elements = [Line2D([0], [0], color=val, label=key) for key,val in color_dict.items()]
            fig.axes[axes_i].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title=color, fontsize=titles_fontsize, title_fontsize=titles_fontsize)

    elif color is not None:
        fig = plt.figure()
        ax = plt.subplot(111)

        for color_val in dt.select(color).unique().to_numpy().flatten():
            color_dt = dt.filter(pl.col(color) == color_val)

            plotter(color_dt, x_name, x_type, fig=fig, label=color_val)

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=color, fontsize=titles_fontsize, title_fontsize=titles_fontsize)

    else:
        fig = plotter(dt, x_name, x_type)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel(x_name, fontsize=titles_fontsize)
    plt.ylabel(y_label, fontsize=titles_fontsize)

    return plt