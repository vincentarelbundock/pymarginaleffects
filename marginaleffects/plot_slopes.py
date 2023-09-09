import polars as pl
import numpy as np

from .utils import get_variable_type, get_modeldata
from .plot_common import dt_on_condition, plot_common
from .slopes import slopes


def plot_slopes(
    model,
    condition=None,
    variables=None,
    newdata=None,
    slope="dydx",
    vcov=True,
    conf_level=0.95,
    by=False,
    wts=None,
    draw=True,
):

    assert not (not by and newdata is not None), "The `newdata` argument requires a `by` argument."

    assert (condition is None and by) or (condition is not None and not by), "One of the `condition` and `by` arguments must be supplied, but not both."

    assert not (wts is not None and not by), "The `wts` argument requires a `by` argument."

    if by:

        if newdata is not None:
            dt = slopes(model,
                variables=variables,
                newdata=newdata,
                slope=slope,
                vcov=vcov,
                conf_level=conf_level,
                by=by,
                wts=wts)
        else:
            dt = slopes(model,
                variables=variables,
                slope=slope,
                vcov=vcov,
                conf_level=conf_level,
                by=by,
                wts=wts)

        var_list = [by] if isinstance(by, str) else by

    elif condition is not None:
        dt_condition = dt_on_condition(model, condition)
        if isinstance(condition, str):
            var_list = [condition]
        elif isinstance(condition, list):
            var_list = condition
        elif isinstance(condition, dict):
            var_list = list(condition.keys())
        dt = slopes(model,
                variables=variables,
                newdata=dt_condition,
                slope=slope,
                vcov=vcov,
                conf_level=conf_level,
                by=var_list,
                wts=wts)

    dt = dt.drop_nulls(var_list[0])
    dt = dt.sort(var_list[0])
    
    if not draw:
        return dt

    color = None
    subplot = None

    if isinstance(variables, list) and len(variables) > 1:
        subplot = 'term'

    if get_variable_type(dt.select(['term']).row(0)[0], get_modeldata(model)) != "numeric":
        assert subplot is None, "Too much variables specified as subplot"
        subplot = 'contrast'

    if len(var_list) == 3:
        assert subplot is None, "Too much variables specified as subplot"
        assert color is None, "Too much variables specified as color"
        return plot_common(dt, "Slope", var_list[0], color=var_list[1], subplot=var_list[2])

    elif len(var_list) == 2:
        return plot_common(dt, "Slope", var_list[0], color=var_list[1], subplot=subplot)

    elif len(var_list) == 1:
        return plot_common(dt, "Slope", var_list[0], color=color, subplot=subplot)

    else:
        raise ArgumentError("Too much variables specified")