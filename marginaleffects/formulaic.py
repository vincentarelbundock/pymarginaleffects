import formulaic
import inspect
import polars as pl
import numpy as np
from .utils import validate_types, ingest


@validate_types
def variables(formula: str):
    """Extract variable names from a model formula.

    Parameters
    ----------
    formula : str
        A model formula string containing '~' to separate dependent and independent variables.
        Example: "y ~ x1 + x2 * x3"

    Returns
    -------
    list
        List of variable names in order of appearance, including both dependent and
        independent variables.

    Raises
    ------
    ValueError
        If formula does not contain '~' separator.

    Examples
    --------
    >>> variables("y ~ x1 + x2")
    ['y', 'x1', 'x2']
    >>> variables("response ~ pred1 * pred2")
    ['response', 'pred1', 'pred2']

    Notes
    -----
    - Uses formulaic's parser to tokenize the formula
    - Only extracts 'name' tokens, ignoring operators and other symbols
    - Returns variables in order of appearance in the formula
    """
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
    """List-wise delete rows with missing values for specified variables.

    Parameters
    ----------
    formula : str, optional
        Model formula from which to extract variable names.
    vars : list[str], optional
        List of variable names to check for missing values.
    data : pl.DataFrame
        Data frame to check for missing values.

    Returns
    -------
    pl.DataFrame
        Data frame with rows removed where specified variables have missing values.

    Raises
    ------
    ValueError
        If neither formula nor vars is provided.

    Notes
    -----
    Either formula or vars must be provided to specify which variables to check.
    """
    if formula is not None:
        vars = variables(formula)
    elif formula is None and vars is None:
        raise ValueError("formula or vars must be provided")
    return data.drop_nulls(subset=vars)


@validate_types
def design(formula: str, data: pl.DataFrame):
    """Create design matrices from formula and data.

    Parameters
    ----------
    formula : str
        Model formula specifying the model structure.
    data : pl.DataFrame
        Data frame containing the variables referenced in the formula.

    Returns
    -------
    tuple
        (y, X, data) where:
            - y : np.ndarray
                Response variable vector
            - X : formulaic.ModelMatrix
                Design matrix for predictors
            - data : pl.DataFrame
                Cleaned data frame with missing values removed

    Notes
    -----
    Converts polars DataFrame to pandas for compatibility with formulaic.
    Flattens y to 1D array if response variable is categorical.
    """
    vars = variables(formula)
    data = data.drop_nulls(subset=vars)
    y, X = formulaic.model_matrix(formula, ingest(data).to_pandas())
    y = np.ravel(data[vars[0]])  # avoid matrix if LHS is a character
    return y, X, data


def _sanity_engine(engine):
    """Validate that an engine object has required fitting capabilities.

    Parameters
    ----------
    engine : object
        Model engine to validate.

    Raises
    ------
    AttributeError
        If engine lacks a fit method.
    ValueError
        If engine.fit doesn't accept X and y parameters.

    Notes
    -----
    Checks both the presence of fit method and its parameter names.
    """
    if not hasattr(engine, "fit"):
        raise AttributeError("engine must have a 'fit' method")
    sig = inspect.signature(engine.fit)
    param_names = list(sig.parameters.keys())
    if "X" not in param_names or "y" not in param_names:
        raise ValueError("engine.fit must accept parameters named 'X' and 'y'")


@validate_types
def fit(formula: str, data: pl.DataFrame, engine):
    """Fit a model using a formula interface.

    Parameters
    ----------
    formula : str
        Model formula specifying the model structure.
    data : pl.DataFrame
        Data frame containing the variables referenced in the formula.
    engine : object
        Model engine with a fit method accepting X and y parameters.

    Returns
    -------
    object
        Fitted model object with additional attributes:
            - formula : str
                The model formula used for fitting
            - data : pl.DataFrame
                The cleaned data used for fitting

    Notes
    -----
    Validates engine, creates design matrices, fits model, and adds attributes.
    """
    _sanity_engine(engine)
    y, X, data = design(formula, data)
    out = engine.fit(X=X, y=y)
    out.formula = formula
    out.data = data
    return out


__all__ = ["variables", "lwd", "design", "fit"]
