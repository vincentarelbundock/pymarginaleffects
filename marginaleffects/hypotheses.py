import numpy as np
import polars as pl

from .hypothesis import get_hypothesis
from .sanity import sanitize_vcov
from .uncertainty import get_jacobian, get_se, get_z_p_ci
from .utils import sort_columns


def hypotheses(model, hypothesis=None, conf_level=0.95, vcov=True):
    # sanity checks
    V = sanitize_vcov(vcov, model)

    # estimands
    def fun(x):
        out = pl.DataFrame({"estimate": x})
        out = get_hypothesis(out, hypothesis=hypothesis)
        return out

    out = fun(np.array(model.params))
    if vcov is not None:
        J = get_jacobian(fun, model.params.to_numpy())
        se = get_se(J, V)
        out = out.with_columns(pl.Series(se).alias("std_error"))
        out = get_z_p_ci(out, model, conf_level=conf_level)
    out = sort_columns(out, by=None)
    return out
