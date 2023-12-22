import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from .datagrid import datagrid  # noqa

from .getters import get_modeldata
from .utils import get_variable_type


def dt_on_condition(model, condition):
    modeldata = get_modeldata(model)

    if isinstance(condition, str):
        condition = [condition]

    assert (
        1 <= len(condition) <= 3
    ), f"Lenght of condition must be inclusively between 1 and 3. Got : {len(condition)}."

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

    for key, value in to_datagrid.items():
        variable_type = get_variable_type(key, modeldata)

        # Check type of user-supplied dict values
        if value is not None:
            test_df = pl.DataFrame({key: value})
            assert (
                variable_type == get_variable_type(key, test_df)
            ), f"Supplied data type of {key} column ({get_variable_type(key, test_df)}) does not match the type of the variable ({variable_type})."
            continue

        if variable_type == "numeric":
            if key == first_key:
                to_datagrid[key] = np.linspace(
                    modeldata[key].min(), modeldata[key].max(), 100
                ).tolist()
            else:
                to_datagrid[key] = np.percentile(
                    modeldata[key], [0, 25, 50, 75, 100], method="midpoint"
                ).tolist()

        elif variable_type == "boolean" or variable_type == "character":
            to_datagrid[key] = modeldata[key].unique().to_list()
            assert (
                len(to_datagrid[key]) <= 10
            ), f"Character type variables of more than 10 unique values are not supported. {key} variable has {len(to_datagrid[key])} unique values."

    dt_code = "datagrid(newdata=modeldata"
    for key, value in to_datagrid.items():
        dt_code += ", " + key + "="
        if isinstance(value, str):
            dt_code += "'" + value + "'"
        else:
            dt_code += str(value)
    dt_code += ")"

    # TODO: this is weird. I'd prefer someting more standard than evaluating text
    exec("global dt; dt = " + dt_code)

    return dt  # noqa: F821


def plotter(dt, x_name, x_type, fig=None, axe=None, label=None, color=None):
    x = dt.select(x_name).to_numpy().flatten()
    y = dt.select("estimate").to_numpy().flatten()
    y_low = dt.select("conf_low").to_numpy().flatten()
    y_high = dt.select("conf_high").to_numpy().flatten()

    if fig is not None:
        plot_obj = fig.axes[axe] if axe is not None else plt
    else:
        fig = plt.figure()
        plot_obj = plt

    if x_type == "numeric":
        if color is None:
            plot_obj.fill_between(x, y_low, y_high, alpha=0.2)
            plot_obj.plot(x, y, label=label)
        else:
            plot_obj.fill_between(x, y_low, y_high, color=color, alpha=0.2)
            plot_obj.plot(x, y, color=color, label=label)

    elif x_type == "character" or x_type == "boolean":
        y_low = np.absolute(y - y_low)
        y_high = np.absolute(y_high - y)
        if color is None:
            plot_obj.errorbar(x, y, yerr=(y_low, y_high), fmt="o", label=label)
        else:
            plot_obj.errorbar(
                x, y, yerr=(y_low, y_high), fmt="o", color=color, label=label
            )

    return fig


def plot_common(dt, y_label, var_list):
    titles_fontsize = 12

    x_name = var_list[0]
    x_type = get_variable_type(x_name, dt)

    if len(var_list) == 2:
        color = var_list[1]
        subplot = [None]
    elif len(var_list) == 3:
        color = var_list[1]
        subplot = (
            dt.select(var_list[2]).unique(maintain_order=True).to_numpy().flatten()
        )
    else:
        color = None
        subplot = [None]

    # when 'contrast' is a column containing more than 1 unique value, we subplot all intersections
    # of these values with explicit subplots
    if "contrast" in dt.columns:
        contrast = (
            dt.select("contrast").unique(maintain_order=True).to_numpy().flatten()
        )
        if len(contrast) == 1:
            contrast = [None]
    else:
        contrast = [None]

    if subplot[0] is not None or contrast[0] is not None:
        color_i = 0
        color_dict = {}

        if len(subplot) >= len(contrast):
            dim_max = subplot
            dim_min = contrast
            max_name = var_list[2] if len(var_list) == 3 else None
            min_name = "contrast"
        else:
            dim_max = contrast
            dim_min = subplot
            max_name = "contrast"
            min_name = var_list[2] if len(var_list) == 3 else None

        max_len = len(dim_max)
        min_len = len(dim_min)

        figsize_def = plt.rcParams.get("figure.figsize")
        figsize = [
            max(figsize_def[0], (2 / 3) * figsize_def[0] * max_len),
            max(figsize_def[1], (2 / 3) * figsize_def[1] * min_len),
        ]

        fig, axes = plt.subplots(
            min_len, max_len, squeeze=False, layout="constrained", figsize=figsize
        )

        for i, dim_min_i in enumerate(dim_min):
            for j, dim_max_j in enumerate(dim_max):
                subplot_dt = dt

                subplot_dt = subplot_dt.filter(pl.col(max_name) == dim_max_j)
                if dim_min_i is not None:
                    subplot_dt = subplot_dt.filter(pl.col(min_name) == dim_min_i)

                axe = max_len * i + j

                if color is None:
                    plotter(subplot_dt, x_name, x_type, fig=fig, axe=axe)

                else:
                    for color_val in (
                        subplot_dt.select(color)
                        .unique(maintain_order=True)
                        .to_numpy()
                        .flatten()
                    ):
                        if color_val not in color_dict:
                            color_dict[color_val] = plt.rcParams[
                                "axes.prop_cycle"
                            ].by_key()["color"][color_i]
                            color_i += 1

                        color_dt = subplot_dt.filter(pl.col(color) == color_val)

                        plotter(
                            color_dt,
                            x_name,
                            x_type,
                            fig=fig,
                            axe=axe,
                            label=color_val,
                            color=color_dict[color_val],
                        )

                if max_name == "contrast":
                    title = dim_min_i if dim_min_i is not None else ""
                    title += (
                        "\n"
                        + subplot_dt.select(pl.first("term")).item()
                        + ", "
                        + dim_max_j
                        if dim_max_j is not None
                        else ""
                    )
                else:
                    title = dim_max_j
                    title += (
                        "\n"
                        + subplot_dt.select(pl.first("term")).item()
                        + ", "
                        + dim_min_i
                        if dim_min_i is not None
                        else ""
                    )

                fig.axes[axe].set_title(title, fontsize=titles_fontsize)

        if color is not None:
            legend_elements = [
                Line2D([0], [0], color=val, label=key)
                for key, val in color_dict.items()
            ]

    elif color is not None:
        fig = plt.figure(layout="constrained")

        for color_val in (
            dt.select(color).unique(maintain_order=True).to_numpy().flatten()
        ):
            color_dt = dt.filter(pl.col(color) == color_val)

            plotter(color_dt, x_name, x_type, fig=fig, label=color_val)

        fig.legend(
            loc="outside center right",
            title=color,
            fontsize=titles_fontsize,
            title_fontsize=titles_fontsize,
        )

    else:
        fig = plotter(dt, x_name, x_type)

    if (subplot[0] is not None or contrast[0] is not None) and color is not None:
        fig.legend(
            handles=legend_elements,
            loc="outside center right",
            title=color,
            fontsize=titles_fontsize,
            title_fontsize=titles_fontsize,
        )
    fig.supxlabel(x_name, fontsize=titles_fontsize)
    fig.supylabel(y_label, fontsize=titles_fontsize)

    return plt
