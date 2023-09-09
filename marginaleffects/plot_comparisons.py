import polars as pl
import numpy as np

from .utils import get_variable_type, get_modeldata
from .plot_common import dt_on_condition, plot_common
from .comparisons import comparisons


def plot_comparisons(
    model,
    condition=None,
    variables=None,
    newdata=None,
    comparison="difference",
    vcov=True,
    conf_level=0.95,
    by=False,
    wts=None,
    draw=True,
    hypothesis=None,
    equivalence=None,
    transform=None,
    eps=1e-4,
):

    assert not (not by and newdata is not None), "The `newdata` argument requires a `by` argument."

    assert (condition is None and by) or (condition is not None and not by), "One of the `condition` and `by` arguments must be supplied, but not both."

    assert not (wts is not None and not by), "The `wts` argument requires a `by` argument."

    if by:

        if newdata is not None:
            dt = comparisons(model,
                    variables=variables,
                    newdata=newdata,
                    comparison=comparison,
                    vcov=vcov,
                    conf_level=conf_level,
                    by=by,
                    wts=wts,
                    hypothesis=hypothesis,
                    equivalence=equivalence,
                    transform=transform,
                    eps=eps)
        else:
            dt = comparisons(model,
                    variables=variables,
                    comparison=comparison,
                    vcov=vcov,
                    conf_level=conf_level,
                    by=by,
                    wts=wts,
                    hypothesis=hypothesis,
                    equivalence=equivalence,
                    transform=transform,
                    eps=eps)

        var_list = [by] if isinstance(by, str) else by

    elif condition is not None:
        dt_condition = dt_on_condition(model, condition)
        if isinstance(condition, str):
            var_list = [condition]
        elif isinstance(condition, list):
            var_list = condition
        elif isinstance(condition, dict):
            var_list = list(condition.keys())
        dt = comparisons(model,
                variables=variables,
                newdata=dt_condition,
                comparison=comparison,
                vcov=vcov,
                conf_level=conf_level,
                by=var_list,
                wts=wts,
                hypothesis=hypothesis,
                equivalence=equivalence,
                transform=transform,
                eps=eps)

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
        return plot_common(dt, "Comparison", var_list[0], color=var_list[1], subplot=var_list[2])

    elif len(var_list) == 2:
        return plot_common(dt, "Comparison", var_list[0], color=var_list[1], subplot=subplot)

    elif len(var_list) == 1:
        return plot_common(dt, "Comparison", var_list[0], color=color, subplot=subplot)

    else:
        raise ArgumentError("Too much variables sepcified")