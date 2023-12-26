# from lets_plot import *
from plotnine import *
import numpy as np
import polars as pl
from .datagrid import datagrid  # noqa
from .sanitize_model import sanitize_model
from .utils import get_variable_type


def plot_common_lp(dt, y_label, var_list):

    if len(var_list) == 1 and "conf_low" in dt.columns:
        vartype = get_variable_type(var_list[0], dt)
        if vartype == "numeric":
            out = (
                ggplot(dt, aes(x=var_list[0], y="estimate", ymin="conf_low", ymax="conf_high")) +
                geom_ribbon(alpha = .2) +
                geom_line() +
                theme_bw()
            )

        else:
            out = (
                ggplot(dt, aes(x=var_list[0], y="estimate", ymin="conf_low", ymax="conf_high")) +
                geom_pointrange() +
                theme_bw()
            )

    elif len(var_list) == 2 and "conf_low" in dt.columns:
        vartype = get_variable_type(var_list[0], dt)
        if vartype == "numeric":
            out = (
                ggplot(dt, aes(x=var_list[0], y="estimate", ymin="conf_low", ymax="conf_high")) +
                geom_ribbon(aes(fill = var_list[1]), alpha = .2) +
                geom_line(aes(color = var_list[1])) +
                theme_bw()
            )

        else:
            out = (
                ggplot(dt, aes(x=var_list[0], y="estimate", ymin="conf_low", ymax="conf_high")) +
                geom_pointrange(aes(color = var_list[1]), position = position_dodge(width = .1)) +
                theme_bw()
            )
    else:
        raise ValueError("Inputs not supported")

    return out