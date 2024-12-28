import formulaic
import polars as pl
from .utils import validate_types


@validate_types
def variables(formula: str):
    tok = formulaic.parser.DefaultFormulaParser().get_tokens(formula)
    tok = [t for t in tok if t.kind.value == "name"]
    tok = [str(t) for t in tok]
    return tok


@validate_types
def listwise_deletion(formula: str, data: pl.DataFrame):
    vars = variables(formula)
    return data.drop_nulls(subset=vars)


def model_matrices(formula: str, data: pl.DataFrame, formula_engine: str = "formulaic"):
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


__all__ = ["listwise_deletion", "model_matrices"]
