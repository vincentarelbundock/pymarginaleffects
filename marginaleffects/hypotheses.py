import numpy as np
import patsy
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .by import *
from .hypothesis import *
from .sanity import *
from .uncertainty import *
from .utils import *


def hypotheses(model, hypothesis=None, conf_int=0.95, vcov=True):
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
        out = get_z_p_ci(out, model, conf_int=conf_int)
    out = sort_columns(out, by=None)
    return out
