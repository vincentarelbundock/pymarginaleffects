from .uncertainty import *
from .sanity import *
from .by import *
from .utils import *
from .hypothesis import *
from .equivalence import *
from .transform import *
import polars as pl
import pandas as pd
import numpy as np
import patsy
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

def get_predictions(model, params, newdata: Union[pl.DataFrame, pd.DataFrame]):
    if isinstance(newdata, np.ndarray):
        exog = newdata
    else:
        y, exog = patsy.dmatrices(model.model.formula, newdata.to_pandas())
    p = model.model.predict(params, exog)
    if p.ndim == 1:
        p = pl.DataFrame({"rowid": range(newdata.shape[0]), "estimate": p})
    elif p.ndim == 2:
        colnames = {f"column_{i}": str(i) for i in range(p.shape[1])}
        p = pl.DataFrame(p) \
            .rename(colnames) \
            .with_columns(pl.Series(range(p.shape[0]), dtype = pl.Int32).alias("rowid")) \
            .melt(id_vars = "rowid", variable_name = "group", value_name = "estimate")
    else:
        raise ValueError("The `predict()` method must return an array with 1 or 2 dimensions.")
    p = p.with_columns(pl.col("rowid").cast(pl.Int32))
    return p


def predictions(
    model,
    conf_int = 0.95,
    vcov = True,
    by = False,
    newdata = None,
    hypothesis = None,
    equivalence = None,
    transform = None):

    # sanity checks
    V = sanitize_vcov(vcov, model)
    newdata = sanitize_newdata(model, newdata)

    # predictors
    y, exog = patsy.dmatrices(model.model.formula, newdata.to_pandas())

    # estimands
    def fun(x):
        out = get_predictions(model, np.array(x), exog)
        out = get_by(model, out, newdata=newdata, by=by)
        out = get_hypothesis(out, hypothesis=hypothesis)
        return out
    out = fun(model.params)
    if vcov is not None:
        J = get_jacobian(fun, model.params)
        se = get_se(J, V)
        out = out.with_columns(pl.Series(se).alias("std_error"))
        out = get_z_p_ci(out, model, conf_int=conf_int)
    out = get_transform(out, transform = transform)
    out = get_equivalence(out, equivalence = equivalence)
    out = sort_columns(out, by = by)
    return out
