import polars as pl
import numpy as np

from .utils import find_response
from .plot_common import dt_on_condition, plot_common
from .predictions import predictions


def plot_predictions(
    model,
    condition=None,
    by=False,
    newdata=None,
    vcov=True,
    conf_level=0.95,
    transform=None,
    draw=True,
    wts=None
):

    assert not (not by and newdata is not None), "The `newdata` argument requires a `by` argument."

    assert (condition is None and by) or (condition is not None and not by), "One of the `condition` and `by` arguments must be supplied, but not both."

    assert not (wts is not None and not by), "The `wts` argument requires a `by` argument."

    if by:

        if newdata is not None:
            dt = predictions(model,
                    by=by,
                    newdata=newdata,
                    conf_level=conf_level,
                    vcov=vcov,
                    transform=transform,
                    wts=wts)
        else:
            dt = predictions(model,
                    by=by,
                    conf_level=conf_level,
                    vcov=vcov,
                    transform=transform,
                    wts=wts)

        var_list = [by] if isinstance(by, str) else by

    if condition is not None:
        dt_condition = dt_on_condition(model, condition)
        if isinstance(condition, str):
            var_list = [condition]
        elif isinstance(condition, list):
            var_list = condition
        elif isinstance(condition, dict):
            var_list = list(condition.keys())
        dt = predictions(model,
                by=var_list,
                newdata=dt_condition,
                conf_level=conf_level,
                vcov=vcov,
                transform=transform)

    dt = dt.drop_nulls(var_list[0])
    dt = dt.sort(var_list[0])
    
    if not draw:
        return dt

    if len(var_list) == 3:
        return plot_common(dt, find_response(model), var_list[0], color=var_list[1], subplot=var_list[2])
    elif len(var_list) == 2:
        return plot_common(dt, find_response(model), var_list[0], color=var_list[1])
    elif len(var_list) == 1:
        return plot_common(dt, find_response(model), var_list[0])
    else:
        raise ArgumentError("Too much variables sepcified")