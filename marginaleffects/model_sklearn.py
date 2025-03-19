import numpy as np
import warnings
import polars as pl
from .docs import DocsModels
from .utils import ingest
from .formulaic_utils import listwise_deletion, model_matrices, get_variables
from .model_abstract import ModelAbstract


class ModelSklearn(ModelAbstract):
    def __init__(self, model):
        if not hasattr(model, "data"):
            raise ValueError("Model must have a 'data' attribute")
        else:
            self.data = ingest(model.data)
        if not hasattr(model, "formula"):
            raise ValueError("Model must have a 'formula' attribute")
        else:
            self.formula = model.formula
        super().__init__(model)

    def get_predict(self, params, newdata: pl.DataFrame):
        if isinstance(newdata, np.ndarray):
            exog = newdata
        else:
            try:
                import formulaic

                if isinstance(newdata, formulaic.ModelMatrix):
                    exog = newdata.to_numpy()
                else:
                    if isinstance(newdata, pl.DataFrame):
                        nd = newdata.to_pandas()
                    else:
                        nd = newdata
                    y, exog = formulaic.model_matrix(self.model.formula, nd)
                    exog = exog.to_numpy()
            except ImportError:
                raise ImportError(
                    "The formulaic package is required to use this feature."
                )

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*valid feature names.*")
                p = self.model.predict_proba(exog)
                # only keep the second column for binary classification since it is redundant info
                if p.shape[1] == 2:
                    p = p[:, 1]
        except (AttributeError, NotImplementedError):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*valid feature names.*")
                p = self.model.predict(exog)

        if p.ndim == 1:
            p = pl.DataFrame({"rowid": range(newdata.shape[0]), "estimate": p})
        elif p.ndim == 2 and p.shape[1] == 1:
            p = pl.DataFrame(
                {"rowid": range(newdata.shape[0]), "estimate": np.ravel(p)}
            )
        elif p.ndim == 2:
            colnames = {f"column_{i}": v for i, v in enumerate(self.model.classes_)}
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

        p = p.with_columns(pl.col("rowid").cast(pl.Int32))

        return p


# @validate_types
def fit_sklearn(
    formula: str, data: pl.DataFrame, engine, kwargs_engine={}, kwargs_fit={}
) -> ModelSklearn:
    """
    Fit a sklearn model with output that is compatible with pymarginaleffects.

    For more information, visit the website: https://marginaleffects.com/

    Or type: `help(fit_sklearn)`
    """
    d = listwise_deletion(formula, data=data)
    y, X = model_matrices(formula, d)
    # formulaic returns a matrix when the response is character or categorical
    if y.ndim == 2:
        y = d[get_variables(formula)[0]]
    y = np.ravel(y)
    out = engine(**kwargs_engine).fit(X=X, y=y, **kwargs_fit)
    out.data = d
    out.formula = formula
    out.formula_engine = "formulaic"
    out.fit_engine = "sklearn"
    return ModelSklearn(out)


docs_sklearn = (
    """
# `fit_sklearn()`

Fit a sklearn model with output that is compatible with pymarginaleffects.

This function streamlines the process of fitting sklearn models by:

1. Parsing the formula
2. Handling missing values
3. Creating model matrices
4. Fitting the model with specified options

## Parameters
"""
    + DocsModels.docstring_formula
    + """
`data`: (pandas.DataFrame) Dataframe with the response variable and predictors.

`engine`: (callable) sklearn model class (e.g., LinearRegression, LogisticRegression)
"""
    + DocsModels.docstring_kwargs_engine
    + """
`kwargs_fit` : (dict, default={}) Additional arguments passed to the model's fit method. 
"""
    + DocsModels.docstring_fit_returns("Sklearn")
    + """
## Examples

```python
from sklearn.linear_model import LinearRegression
from marginaleffects import *

data = get_dataset("thornton")

model = fit_sklearn(
    formula="outcome ~ distance + incentive",
    data=data,
    engine=LinearRegression,
)

predictions(model)
```
"""
    + DocsModels.docstring_notes("sklearn")
)

fit_sklearn.__doc__ = docs_sklearn
