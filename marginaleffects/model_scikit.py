import re
import numpy as np
import polars as pl
import warnings
import formulaic
from .model_abstract import ModelAbstract


class ModelScikitLogisticRegression(ModelAbstract):
    def __init__(self, model):
        super().__init__(model)
        self.formula = model.formula
        self.data = model.data

    def get_variables_names(self, variables=None, newdata=None):
        if variables is None:
            formula = self.model.formula
            columns = self.model.data.columns
            order = {}
            for var in columns:
                match = re.search(rf"\b{re.escape(var)}\b", formula.split("~")[1])
                if match:
                    order[var] = match.start()
            variables = sorted(order, key=lambda i: order[i])

        if isinstance(variables, (str, dict)):
            variables = [variables] if isinstance(variables, str) else variables
        elif isinstance(variables, list) and all(
            isinstance(var, str) for var in variables
        ):
            pass
        else:
            raise ValueError(
                "`variables` must be None, a dict, string, or list of strings"
            )

        if newdata is not None:
            good = [x for x in variables if x in newdata.columns]
            bad = [x for x in variables if x not in newdata.columns]
            if len(bad) > 0:
                bad = ", ".join(bad)
                warnings.warn(f"Variable(s) not in newdata: {bad}")
            if len(good) == 0:
                raise ValueError("There is no valid column name in `variables`.")
        return variables

    def get_predict(self, params, newdata: pl.DataFrame):
        if isinstance(newdata, np.ndarray):
            exog = newdata
        elif isinstance(newdata, formulaic.ModelMatrix):
            exog = newdata.to_numpy()
        else:
            if isinstance(newdata, pl.DataFrame):
                nd = newdata.to_pandas()
            else:
                nd = newdata
            y, exog = formulaic.model_matrix(self.model.formula, nd)
            exog = exog.to_numpy()
        p = self.model.predict_proba(exog)[:, 1]
        p = pl.DataFrame({"rowid": range(len(p)), "estimate": p})
        p = p.with_columns(pl.col("rowid").cast(pl.Int32))
        return p

    def get_coef(self):
        return np.array([])
    
    def get_coef_names(self):
        return None

    def get_response_name(self):
        return ""