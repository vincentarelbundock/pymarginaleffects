import formulaic
import inspect
import polars as pl
import numpy as np


def get_variables(formula: str):
    if "~" not in formula:
        raise ValueError("formula must contain '~' to separate dependent and independent variables")
    tok = formulaic.parser.DefaultFormulaParser().get_tokens(formula)
    tok = [t for t in tok if t.kind.value == "name"]
    tok = [str(t) for t in tok]
    return tok


def fit(formula: str, data: pl.DataFrame, engine):
    # Validate engine
    if not hasattr(engine, 'fit'):
        raise AttributeError("engine must have a 'fit' method")
    sig = inspect.signature(engine.fit)
    param_names = list(sig.parameters.keys())
    if 'X' not in param_names or 'y' not in param_names:
        raise ValueError("engine.fit must accept parameters named 'X' and 'y'")

    # list-wise deletion
    vars = get_variables(formula)
    data = data.drop_nulls(subset=vars)

    # Fit the model and add attributes
    y, X = formulaic.model_matrix(formula, data.to_pandas())
    if y.ndim == 2:
        y = data[vars[0]]
    y = np.ravel(y)

    out = engine.fit(X = X, y = y)

    out.formula = formula
    out.data = data
    out.variables = vars

    return out

