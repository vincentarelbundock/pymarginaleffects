from plotnine import *
import numpy as np
import polars as pl
from .datagrid import datagrid  # noqa
from .sanitize_model import sanitize_model
from .utils import get_variable_type


def plot_common(dt, y_label, var_list):

    discrete = get_variable_type(var_list[0], dt) != "numeric"
    interval = "conf_low" in dt.columns

    # aes
    mapping = {"x": var_list[0], "y": "estimate"}
    if interval:
        mapping["ymin"] = "conf_low"
        mapping["ymax"] = "conf_high"
    if len(var_list) > 1:
        mapping["color"] = var_list[1]
        mapping["fill"] = var_list[1]
    mapping = aes(**mapping)

    p = ggplot(data = dt, mapping = mapping)

    if discrete:
        if interval:
            if len(var_list) > 1:
                p = p + geom_pointrange(position = position_dodge(width = .1))
            else:
                p = p + geom_pointrange()
        else:
            p = p + geom_point()
    else:
        if interval:
            p = (
                p +
                geom_ribbon(alpha = .2) +
                geom_line()
            )
        else:
            p = p + geom_line()

    if len(var_list) == 3:
        p = p + facet_wrap(f"~ {var_list[2]}")

    elif len(var_list) == 4:
        p = p + facet_grid(f"{var_list[2]} ~ {var_list[3]}")

    return p