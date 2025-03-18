import re
import numpy as np
import pandas as pd
import narwhals as nw
from typing import Any, Dict
import polars as pl
from .docs import DocsModels
from .utils import ingest
from .model_abstract import ModelAbstract
from .formulaic_utils import (
    listwise_deletion,
    model_matrices,
    parse_linearmodels_formula,
)


class ModelLinearmodels(ModelAbstract):
    """
    Interface between linearmodels and marginaleffects for panel models.

    This class handles the conversion between linearmodels' MultiIndex pandas
    DataFrames and marginaleffects' polars DataFrames. It ensures proper data
    structure handling and index preservation across the two frameworks.

    Parameters
    ----------
    model : linearmodels.panel.results.PanelResults
        A fitted linearmodels panel model. Must have 'data' and 'formula' attributes.


    Attributes
    ----------
    multiindex_names : list[str]
        Names of the MultiIndex levels from the original data.
    data : polars.DataFrame
        The model data converted to polars format.
    formula : str
        The model formula used in estimation.

    Raises
    ------
    ValueError
        If the model lacks required 'data' or 'formula' attributes.

      Examples
    --------
    >>> import pandas as pd
    >>> from linearmodels.panel import PanelOLS
    >>> formula = "y ~ x1 + EntityEffects"
    >>> model = fit_linearmodels(
    ...     formula=formula,
    ...     data=data,
    ...     engine=PanelOLS,
    ...     kwargs_fit={'cov_type': 'robust'}
    ... )

    Notes
    -----
    The class maintains the mapping between index variables in the original
    pandas DataFrame and their column representation in polars, ensuring
    consistent data manipulation across frameworks.

    See Also
    --------
    fit_linearmodels : Helper function to create ModelLinearmodels instances
    """

    def __init__(self, model):
        if not hasattr(model, "data"):
            raise ValueError("Model must have a 'data' attribute")
        else:
            self.multiindex_names = list(model.data.index.names)
            self.data = ingest(model.data)
        if not hasattr(model, "formula"):
            raise ValueError("Model must have a 'formula' attribute")
        else:
            self.formula = model.formula

        self.initialized_engine = model.initialize_engine
        super().__init__(model)

    def _to_pandas(self, df):
        """
        Convert a DataFrame to pandas format with MultiIndex.

        Transforms a DataFrame containing index columns into a pandas DataFrame
        with these columns set as MultiIndex levels.

        Parameters
        ----------
        df : nw.IntoFrame
            DataFrame containing the original index variables as columns.
            Must include all columns specified in self.multiindex_names.

        Returns
        -------
        pandas.DataFrame
            DataFrame with MultiIndex constructed from the index columns.

        Raises
        ------
        ValueError
            If any of the required index columns are missing from the input DataFrame.
        """

        if not set(self.multiindex_names).issubset(nw.from_native(df).columns):
            raise ValueError(
                f"The DataFrame must contain the original multiindex ({','.join(self.multiindex_names)}) as columns."
            )

        return nw.from_native(df).to_pandas().set_index(self.multiindex_names)

    def get_coef(self):
        return np.array(self.model.params)

    def get_coef_names(self):
        return np.array(self.model.params.index.to_numpy())

    def get_modeldata(self):
        return self._pd_to_pl(self.data)

    def get_vcov(self, vcov=True):
        if isinstance(vcov, bool):
            if vcov is True:
                V = self.model.cov
            else:
                V = None

        if isinstance(vcov, str):
            supported_vcov = [
                "unadjusted",
                "homoskedastic",
                "robust",
                "heteroskedastic",
                "driscoll-kraay",
                "autocorrelated",
                "cluster",
                "kernel",
            ]
            if vcov not in supported_vcov:
                raise ValueError(
                    f"Unknown vcov type: {vcov}.\n"
                    f"Valid options are: {', '.join(supported_vcov)}"
                )

            V = self.initialized_engine.fit(cov_type=vcov).cov

        if V is not None:
            V = np.array(V)
            if V.shape != (len(self.coef), len(self.coef)):
                raise ValueError(
                    "vcov must be a square numpy array with dimensions equal to the length of self.coef"
                )

        return V

    def find_response(self):
        return self.model.model.dependent.vars[0]

    def find_predictors(self):
        formula = self.formula
        columns = self.data.columns
        order = {}
        for var in columns:
            match = re.search(rf"\b{re.escape(var)}\b", formula.split("~")[1])
            if match:
                order[var] = match.start()
        variables = sorted(order, key=lambda i: order[i])
        return variables

    def get_predict(self, params, newdata):
        if isinstance(newdata, np.ndarray):
            exog = newdata
        else:
            y, exog = model_matrices(
                self.formula, self._to_pandas(newdata), formula_engine="linearmodels"
            )

        p = self.model.model.predict(params=params, exog=exog).predictions.values

        if p.ndim == 1:
            p = pl.DataFrame({"rowid": range(newdata.shape[0]), "estimate": p})

        elif p.ndim == 2:
            colnames = {f"column_{i}": str(i) for i in range(p.shape[1])}
            p = (
                pl.DataFrame(p)
                .rename(colnames)
                .with_columns(
                    pl.Series(range(p.shape[0]), dtype=pl.Int32).alias("rowid")
                )
                .melt(id_vars="rowid", variable_name="group", value_name="estimate")
            )
        else:
            raise ValueError(
                "The `predict()` method must return an array with 1 or 2 dimensions."
            )

        return p.with_columns(pl.col("rowid").cast(pl.Int32))

    def get_df(self):
        return self.model.df_resid


# @validate_types
def fit_linearmodels(
    formula: str,
    data: pd.DataFrame,
    engine: None,
    kwargs_engine: Dict[str, Any] = {},
    kwargs_fit: Dict[str, Any] = {},
) -> ModelLinearmodels:
    """
    Fit a linearmodels model with output that is compatible with pymarginaleffects.

    For more information, visit the website: https://marginaleffects.com/

    Or type: `help(fit_linearmodels)`
    """
    linearmodels_formula, effects = parse_linearmodels_formula(formula)

    d = listwise_deletion(linearmodels_formula, data=data)
    y, X = model_matrices(linearmodels_formula, d, formula_engine="linearmodels")
    initialized_engine = engine(dependent=y, exog=X, **kwargs_engine, **effects)

    out = initialized_engine.fit(**kwargs_fit)

    out.data = d
    out.formula = linearmodels_formula
    out.formula_engine = "linearmodels"
    out.initialize_engine = initialized_engine
    out.fit_engine = "linearmodels"

    return ModelLinearmodels(out)


docs_linearmodels = (
    """
# `fit_linearmodels()`

Fit a linearmodels model with output that is compatible with pymarginaleffects.

This function streamlines the process of fitting linearmodels panel models by:

1. Parsing panel effects from the formula
2. Handling missing values
3. Creating model matrices
4. Fitting the model with specified options

## Parameters

`formula`: (str) Model formula with optional panel effects terms. 

- Supported effects are:
    - EntityEffects: Entity-specific fixed effects
    - TimeEffects: Time-specific fixed effects
    - FixedEffects: Alias for EntityEffects
- Example: `"y ~ x1 + x2 + EntityEffects"`

`data` : (pandas.DataFrame) Panel data with MultiIndex (entity, time) or regular DataFrame with entity and time columns.

`engine`: (callable) linearmodels model class (e.g., PanelOLS, BetweenOLS, FirstDifferenceOLS)

`kwargs_engine`: (dict, default={}) Additional arguments passed to the model initialization.

* Example: `{'weights': weights_array}`

`kwargs_fit`: (dict, default={}) Additional arguments passed to the model's fit method.

* Example: `{'cov_type': 'robust'}`
"""
    + DocsModels.docstring_fit_returns("Linearmodels")
    + """
## Examples

```python
from linearmodels.panel import PanelOLS
from linearmodels.panel import generate_panel_data
from marginaleffects import *
data = generate_panel_data()
model_robust = fit_linearmodels(
    formula="y ~ x1 + EntityEffects",
    data=data.data,
    engine=PanelOLS,
    kwargs_fit={'cov_type': 'robust'}
)

predictions(model_robust)
```
"""
    + DocsModels.docstring_notes("linearmodels")
)

fit_linearmodels.__doc__ = docs_linearmodels
