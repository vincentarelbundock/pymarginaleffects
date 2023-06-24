from .uncertainty import *
from .sanity import *
from .by import *
from .utils import *
from .hypothesis import *
import polars as pl
import pandas as pd
import numpy as np
import patsy
import statsmodels.formula.api as smf
import statsmodels.api as sm

def hypotheses(fit, conf_int = 0.95, vcov = True, hypothesis = None):
    # sanity checks
    V = sanitize_vcov(vcov, fit)
    # estimands
    def fun(x):
        out = pl.DataFrame({"estimate": fit.params})
        out = get_hypothesis(out, hypothesis=hypothesis)
        return out
    out = fun(np.array(fit.params))
    if vcov is not None:
        J = get_jacobian(fun, fit.params.to_numpy())
        se = get_se(J, V)
        out = out.with_columns(pl.Series(se).alias("std_error"))
        out = get_z_p_ci(out, fit, conf_int=conf_int)
    out = sort_columns(out, by = None)
    return out

