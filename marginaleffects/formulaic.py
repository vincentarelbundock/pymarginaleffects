import formulaic
import narwhals as nw
import numpy as np
import pandas as pd
from narwhals.typing import IntoFrame

__all__ = ["listwise_deletion", "model_matrices"]


# @validate_types
def extract_variables(formula: str) -> list[str]:
    """
    Extract all variables (column names) from a formula string.

    Parameters
    ----------
    formula : str
        A string representing a statistical formula (e.g., "y ~ x1 + x2").

    Returns
    -------
    list[str]
        A list of variable names extracted from the formula. Only names/identifiers
        are included, operators and special tokens are excluded.

    Examples
    --------
    >>> extract_variables("y ~ x1 + x2")
    ['y', 'x1', 'x2']
    """
    tokens = formulaic.parser.DefaultFormulaParser().get_tokens(formula)

    return [str(token) for token in tokens if token.kind.value == "name"]


# @validate_types
def listwise_deletion(formula: str, data: "IntoFrame"):
    """
    Remove all rows with missing values in any of the variables specified in the formula.

    Parameters
    ----------
    formula : str
        A string representing a statistical formula (e.g., "y ~ x1 + x2") from which
        variable names will be extracted.
    data : IntoFrame
        The input data frame containing the variables. Can be any type that can be
        converted to a native data frame (pandas DataFrame, polars DataFrame, etc.).

    Returns
    -------
    IntoFrame
        A new data frame of the same type as the input, with rows removed where
        any of the variables in the formula contain missing values.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'y': [1, 2, None, 4],
    ...     'x1': [1, None, 3, 4],
    ...     'x2': [1, 2, 3, 4]
    ... })
    >>> listwise_deletion("y ~ x1 + x2", df)
       y  x1  x2
    0  1   1   1
    3  4   4   4
    """
    variables = extract_variables(formula)
    data = nw.from_native(data)
    return data.drop_nulls(subset=variables).to_native()


def parse_linearmodels_formula(formula: str):
    """
    Parse a formula as linearmodels would and extract panel effects specifications.

    This function processes a formula containing potential EntityEffects, FixedEffects,
    and TimeEffects terms. It removes these effect terms from the formula and converts
    them into keyword arguments for linearmodels estimation functions.

    Parameters
    ----------
    formula : str
        A string representing a linearmodels formula (e.g., "y ~ x1 + x2 + EntityEffects").
        The formula may contain special terms: EntityEffects, FixedEffects, and TimeEffects.

    Returns
    -------
    tuple[str, dict[str, bool]]
        A tuple containing:
        - str: The cleaned formula with effects terms removed
        - dict: Keyword arguments for panel effects with keys:
            - 'entity_effects': True if EntityEffects or FixedEffects present
            - 'time_effects': True if TimeEffects present

    Raises
    ------
    ValueError
        If both EntityEffects and FixedEffects are present in the formula.

    Examples
    --------
    >>> formula = "y ~ x1 + FixedEffects"
    >>> parse_linearmodels_formula(formula)
    ('y ~ x1', {'entity_effects': True, 'time_effects': False})

    Notes
    -----
    - EntityEffects and FixedEffects are treated as equivalent for entity effects
    - The function assumes the first variable in the formula is the dependent variable
    - The returned formula will be in the format "y ~ x1 + x2 + ..."
    """

    effects_tokens = {
        "EntityEffects": False,
        "FixedEffects": False,
        "TimeEffects": False,
    }
    effects_kwargs = {"entity_effects": False, "time_effects": False}

    variables = extract_variables(formula)

    for effect in effects_tokens.keys():
        if effect in variables:
            effects_tokens[effect] = True
            variables.remove(effect)

    if effects_tokens["EntityEffects"] and effects_tokens["FixedEffects"]:
        raise ValueError("Cannot use both FixedEffects and EntityEffects")

    effects_kwargs["entity_effects"] = (
        effects_tokens["EntityEffects"] or effects_tokens["FixedEffects"]
    )
    effects_kwargs["time_effects"] = effects_tokens["TimeEffects"]

    cleaned_formula = f"{variables[0]} ~ {' + '.join(variables[1:])}"

    return cleaned_formula, effects_kwargs


def model_matrices(
    formula: str, data: "IntoFrame", formula_engine: str = "formulaic"
) -> tuple[np.ndarray | None | pd.DataFrame, np.ndarray | pd.DataFrame]:
    """
    Construct model matrices (design matrices) from a formula and data using different formula engines.

    Parameters
    ----------
    formula : str
        A string specifying the model formula. The format depends on the formula_engine:
        - formulaic: "y ~ x1 + x2"
        - patsy: "x1 + x2" (right-hand side only)
        - linearmodels: "y ~ x1 + x2 + EntityEffects"
    data : IntoFrame
        The input data frame containing the variables. Can be any type that can be
        converted to a pandas DataFrame.
    formula_engine : str, default="formulaic"
        The formula processing engine to use. Options are:
        - "formulaic": Uses the formulaic package
        - "patsy": Uses the patsy package
        - "linearmodels": Parses formulas with formulaic as linearmodels would

    Returns
    -------
    tuple[np.ndarray | None | pd.DataFrame, np.ndarray | pd.DataFrame]
        A tuple containing:
        - First element: Endogenous variable matrix (dependent variable)
            - numpy array for formulaic
            - None for patsy
            - pandas DataFrame for linearmodels
        - Second element: Exogenous variable matrix (independent variables)
            - numpy array for formulaic
            - numpy array for patsy
            - pandas DataFrame for linearmodels
    """
    data = nw.from_native(data)

    if formula_engine == "formulaic":
        endog, exog = formulaic.model_matrix(formula, data.to_pandas())
        return endog.to_numpy(), exog.to_numpy()

    elif formula_engine == "patsy":
        try:
            import patsy
        except ImportError:
            raise ImportError("The patsy package is required to use this feature.")

        if isinstance(formula, str):
            formula = formula.split("~")[1].strip()

        exog = patsy.dmatrix(formula, data.to_pandas())
        return None, exog

    elif formula_engine == "linearmodels":
        linearmodels_formula, _ = parse_linearmodels_formula(formula)
        endog, exog = formulaic.Formula.get_model_matrix(
            linearmodels_formula, data.to_pandas()
        )
        return pd.DataFrame(endog), pd.DataFrame(exog)
