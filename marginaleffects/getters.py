import re

import numpy as np
import polars as pl
import warnings
import patsy


def get_modeldata(fit):
    df = fit.model.data.frame
    try:
        out = pl.from_pandas(df)
    except:  # noqa: E722
        out = df
    return out


def find_response(model):
    return model.model.endog_names


def get_coef(model):
    return np.array(model.params)


def get_vcov(model, vcov=True):
    if isinstance(vcov, bool):
        if vcov is True:
            V = model.cov_params()
        else:
            V = None

    elif isinstance(vcov, str):
        lab = f"cov_{vcov}"
        if hasattr(model, lab):
            V = getattr(model, lab)
        else:
            raise ValueError(f"The model object has no {lab} attribute.")

    else:
        raise ValueError(
            '`vcov` must be a boolean or a string like "HC3", which corresponds to an attribute of the model object such as "vcov_HC3".'
        )

    V = np.array(V)

    return V


def get_variables_names(variables, model, newdata):
    if variables is None:
        variables = model.model.exog_names
        variables = [re.sub("\[.*\]", "", x) for x in variables]
        variables = [x for x in variables if x in newdata.columns]
        variables = pl.Series(variables).unique().to_list()
    elif isinstance(variables, str):
        variables = [variables]
    else:
        assert isinstance(
            variables, dict
        ), "`variables` must be None, a dict, string, or list of strings"
    good = [x for x in variables if x in newdata.columns]
    bad = [x for x in variables if x not in newdata.columns]
    if len(bad) > 0:
        bad = ", ".join(bad)
        warnings.warn(f"Variable(s) not in newdata: {bad}")
    if len(good) == 0:
        raise ValueError("There is no valid column name in `variables`.")
    return variables


def get_predict(model, params, newdata: pl.DataFrame):
    if isinstance(newdata, np.ndarray):
        exog = newdata
    else:
        y, exog = patsy.dmatrices(model.model.formula, newdata.to_pandas())
    p = model.model.predict(params, exog)
    if p.ndim == 1:
        p = pl.DataFrame({"rowid": range(newdata.shape[0]), "estimate": p})
    elif p.ndim == 2:
        colnames = {f"column_{i}": str(i) for i in range(p.shape[1])}
        p = (
            pl.DataFrame(p)
            .rename(colnames)
            .with_columns(pl.Series(range(p.shape[0]), dtype=pl.Int32).alias("rowid"))
            .melt(id_vars="rowid", variable_name="group", value_name="estimate")
        )
    else:
        raise ValueError(
            "The `predict()` method must return an array with 1 or 2 dimensions."
        )
    p = p.with_columns(pl.col("rowid").cast(pl.Int32))
    return p
