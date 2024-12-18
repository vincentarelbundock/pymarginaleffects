import re
import numpy as np
import polars as pl
import patsy
from .model_abstract import ModelAbstract
from .utils import ingest


class ModelStatsmodels(ModelAbstract):
    def __init__(self, model):
        self.formula = model.model.formula
        self.data = ingest(model.model.data.frame)
        super().__init__(model)
        self.formula_engine = "patsy"
        self.design_info_patsy = model.model.data.design_info

    def get_coef(self):
        return np.array(self.model.params)

    def find_coef(self):
        return np.array(self.model.params.index.to_numpy())

    def get_modeldata(self):
        return self.data

    def get_vcov(self, vcov=True):
        if isinstance(vcov, bool):
            if vcov is True:
                V = self.model.cov_params()
            else:
                V = None
        elif isinstance(vcov, str):
            lab = f"cov_{vcov}"
            if hasattr(self.model, lab):
                V = getattr(self.model, lab)
            else:
                raise ValueError(f"The model object has no {lab} attribute.")
        else:
            raise ValueError(
                '`vcov` must be a boolean or a string like "HC3", which corresponds to an attribute of the model object such as "vcov_HC3".'
            )

        if V is not None:
            V = np.array(V)
            if V.shape != (len(self.coef), len(self.coef)):
                raise ValueError(
                    "vcov must be a square numpy array with dimensions equal to the length of self.coef"
                )

        return V

    def find_variables(self):
        formula = self.formula
        columns = self.modeldata.columns
        order = {}
        for var in columns:
            match = re.search(rf"\b{re.escape(var)}\b", formula.split("~")[1])
            if match:
                order[var] = match.start()
        variables = sorted(order, key=lambda i: order[i])
        return variables

    def get_predict(self, params, newdata: pl.DataFrame):
        if isinstance(newdata, np.ndarray):
            exog = newdata
        elif hasattr(newdata, "to_numpy"):
            exog = newdata.to_numpy()
        else:
            newdata = newdata.to_pandas()
            y, exog = patsy.dmatrices(self.formula, newdata)
        p = self.model.model.predict(params, exog)
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
        p = p.with_columns(pl.col("rowid").cast(pl.Int32))
        return p

    def get_df(self):
        return self.model.df_resid
