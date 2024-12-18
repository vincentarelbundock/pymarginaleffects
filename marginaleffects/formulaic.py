import formulaic
import inspect
import polars as pl
import numpy as np
from .utils import validate_types, ingest


@validate_types
def variables(formula: str):
    if "~" not in formula:
        raise ValueError(
            "formula must contain '~' to separate dependent and independent variables"
        )
    tok = formulaic.parser.DefaultFormulaParser().get_tokens(formula)
    tok = [t for t in tok if t.kind.value == "name"]
    tok = [str(t) for t in tok]
    return tok


@validate_types
def lwd(
    formula: str | None = None, vars: list[str] | None = None, data: pl.DataFrame = None
):
    if formula is not None:
        vars = variables(formula)
    elif formula is None and vars is None:
        raise ValueError("formula or vars must be provided")
    return data.drop_nulls(subset=vars)


def model_matrices(formula: str, data: pl.DataFrame, formula_engine="formulaic"):
    if formula_engine == "formulaic":
        endog, exog = formulaic.model_matrix(formula, data.to_pandas())
        endog = endog.to_numpy()
        exog = exog.to_numpy()
        return endog, exog
    elif formula_engine == "patsy":
        try:
            import patsy
        except ImportError:
            raise ImportError("The patsy package is required to use this feature.")
        if isinstance(formula, str):
            import re

            formula = re.sub(".*~", "", formula)
        exog = patsy.dmatrix(formula, data.to_pandas())
        return None, exog


@validate_types
def design(formula: str, data: pl.DataFrame):
    vars = variables(formula)
    data = data.drop_nulls(subset=vars)
    y, X = formulaic.model_matrix(formula, ingest(data).to_pandas())
    y = np.ravel(data[vars[0]])  # avoid matrix if LHS is a character
    return y, X, data


def _sanity_engine(engine):
    if not hasattr(engine, "fit"):
        raise AttributeError("engine must have a 'fit' method")
    sig = inspect.signature(engine.fit)
    param_names = list(sig.parameters.keys())
    if "X" not in param_names or "y" not in param_names:
        raise ValueError("engine.fit must accept parameters named 'X' and 'y'")


@validate_types
def fit(formula: str, data: pl.DataFrame, engine):
    _sanity_engine(engine)
    y, X, data = design(formula, data)
    out = engine.fit(X=X, y=y)
    out.formula = formula
    out.data = data
    return out


__all__ = ["variables", "lwd", "design", "fit"]
