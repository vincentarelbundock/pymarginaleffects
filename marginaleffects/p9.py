from plotnine import (
    aes,
    facet_wrap,
    facet_grid,
    geom_pointrange,
    geom_ribbon,
    geom_line,
    geom_point,
    ggplot,
    labs,
    position_dodge,
    scale_fill_grey,
    scale_linetype_manual,
)
import polars as pl


def plot_common(model, dt, y_label, var_list, gray=False):
    discrete = model.get_variable_type()[var_list[0]] not in ["numeric", "integer"]
    interval = "conf_low" in dt.columns

    # treat all variables except x-axis as categorical
    if len(var_list) > 1:
        for i in range(1, len(var_list)):
            if dt[var_list[i]].dtype.is_numeric() and i != 0 and i != 1:
                dt = dt.with_columns(pl.col(var_list[i]))
            elif dt[var_list[i]].dtype != pl.Categorical:
                dt = dt.with_columns(pl.col(var_list[i]).cast(pl.Utf8))

    # aes
    # mapping = {"x": var_list[0], "y": y_label}  # proposed change to make y axis label correspond to R  but needs some debugging
    mapping = {"x": var_list[0], "y": "estimate"}
    if interval:
        mapping["ymin"] = "conf_low"
        mapping["ymax"] = "conf_high"
    mapping = aes(**mapping)

    p = ggplot(data=dt, mapping=mapping)

    if discrete:
        if interval:
            if len(var_list) > 1:  #
                p = p + geom_pointrange(
                    aes(shape=var_list[1])
                    if gray
                    else aes(color=var_list[1]),  # so this works correctly for gray
                    position=position_dodge(width=0.1),
                )
            else:
                p = (
                    p + geom_pointrange()
                )  # this does not need grayscale as it does not display colors anyways, run  discrete_interval_len1 to see
        else:
            p = (
                p + geom_point()
            )  # this does not need grayscale as it does not display colors anyways, run  discrete_not_interval to see
    else:
        if interval:
            if len(var_list) > 1:
                p = p + geom_ribbon(
                    aes(
                        fill=var_list[1]
                    ),  # fixed, run experiment not_discrete_interval_len2 to see
                    alpha=0.2,
                )
                if gray:
                    p = p + scale_fill_grey(
                        start=0.2, end=0.8
                    )  # this could be improved by putting texture on the background
            else:
                p = (
                    p + geom_ribbon(alpha=0.2)
                )  # this does not need grayscale as it does not display colors anyways, run  not_discrete_interval_len1 to see
        if len(var_list) > 1:  # we are here <----
            if gray:
                if len(var_list[1]) > 5:
                    raise ValueError(
                        f"The number of elements in the second position of the `condition` or `by` argument (variable {var_list[1]}) cannot exceed 5. It has currently {len(var_list[1])} elements."
                    )
                custom_line_types = [
                    "solid",
                    "dashed",
                    "dotted",
                    "dashdot",
                    (2, (5, 3, 1, 3, 1, 3)),
                ]  # maximum number of lines is 5, this is the default, can add more linetypes by following the documentation at https://plotnine.org/reference/scale_linetype_manual.html
                p = p + geom_line(aes(linetype=var_list[1]))
                p = p + scale_linetype_manual(values=custom_line_types)
            else:
                p = p + geom_line(aes(color=var_list[1]))
        else:
            p = (
                p + geom_line()
            )  # this does not need grayscale as it does not display colors anyways, run  not_discrete_interval_len1 to see

    if len(var_list) == 3:
        p = p + facet_wrap(f"~ {var_list[2]}")

    elif len(var_list) == 4:
        p = p + facet_grid(f"{var_list[3]} ~ {var_list[2]}", scales="free")

    p = p + labs(y=y_label)

    return p
