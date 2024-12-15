import numpy as np
import warnings
import polars as pl
from .utils import formula_to_variables
from .model_abstract import ModelAbstract


class ModelScikitLogisticRegression(ModelAbstract):
    def __init__(self, model):
        super().__init__(model)
        self.formula = model.formula

        # Validate data attribute
        if not hasattr(model, 'data'):
            raise ValueError("Model must have a 'data' attribute")
        if not isinstance(model.data, pl.DataFrame):
            raise TypeError("Model data must be a polars DataFrame")
        self.data = model.data

    def get_variables_names(self, variables=None, newdata=None):
        out = formula_to_variables(self.formula, self.data)
        return out

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
                raise ImportError("The formulaic package is required to use this feature.")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*valid feature names.*")
            p = self.model.predict_proba(exog)[:, 1]

        p = pl.DataFrame({"rowid": range(len(p)), "estimate": p})
        p = p.with_columns(pl.col("rowid").cast(pl.Int32))
        return p