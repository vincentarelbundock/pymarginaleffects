from plotnine import (
    ggplot,
    aes,
    geom_pointrange,
    position_dodge,
    geom_ribbon,
    geom_line,
    geom_point,
    facet_wrap,
    facet_grid,
)
import polars as pl
from .utils import get_variable_type


def plot_common(dt, y_label, var_list):
    discrete = get_variable_type(var_list[0], dt) != "numeric"
    interval = "conf_low" in dt.columns

    # treat all variables except x-axis as categorical
    if len(var_list) > 1:
        for i in range(1, len(var_list)):
            if dt[var_list[i]].dtype != pl.Categorical:
                dt = dt.with_columns(pl.col(var_list[i]).cast(pl.Utf8))

    # aes
    mapping = {"x": var_list[0], "y": "estimate"}
    if interval:
        mapping["ymin"] = "conf_low"
        mapping["ymax"] = "conf_high"
    mapping = aes(**mapping)

    p = ggplot(data=dt, mapping=mapping)

    if discrete:
        if interval:
            if len(var_list) > 1:
                p = p + geom_pointrange(
                    aes(color=var_list[1]), position=position_dodge(width=0.1)
                )
            else:
                p = p + geom_pointrange()
        else:
            p = p + geom_point()
    else:
        if interval:
            if len(var_list) > 1:
                p = p + geom_ribbon(aes(fill=var_list[1]), alpha=0.2)
            else:
                p = p + geom_ribbon(alpha=0.2)
        if len(var_list) > 1:
            p = p + geom_line(aes(color=var_list[1]))
        else:
            p = p + geom_line()

    if len(var_list) == 3:
        p = p + facet_wrap(f"~ {var_list[2]}")

    elif len(var_list) == 4:
        p = p + facet_grid(f"{var_list[2]} ~ {var_list[3]}")

    return p
