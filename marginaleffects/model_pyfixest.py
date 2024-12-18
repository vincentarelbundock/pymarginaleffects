import re
import numpy as np
import polars as pl
from .model_abstract import ModelAbstract
from .utils import ingest


class ModelPyfixest(ModelAbstract):
    def __init__(self, model):
        self.data = ingest(model._data)
        self.formula = model._fml
        super().__init__(model)
        if hasattr(self.model, "_fixef"):
            if self.model._fixef is not None:
                fe = self.model._fixef.split("+")
                for f in fe:
                    self.variables_type[f] = "character"

    def get_coef(self):
        return np.array(self.model._beta_hat)

    def find_coef(self):
        return np.array(self.model._coefnames)

    def get_vcov(self, vcov=True):
        V = None
        if isinstance(vcov, bool):
            if vcov is True:
                V = self.model._vcov
        return V

    def find_predictors(self):
        variables = self.model._coefnames
        variables = [re.sub(r"\[.*\]", "", x) for x in variables]
        variables = [x for x in variables if x in self.data.columns]
        variables = pl.Series(variables).unique().to_list()
        return variables

    def get_predict(self, params, newdata: pl.DataFrame):
        # override the coefficients inside the model object to make different
        # predictions
        m = self.model
        m._beta_hat = params

        # pyfixest does not support polars
        try:
            newdata = newdata.to_pandas()
        except:  #  noqa
            pass

        p = m.predict(newdata=newdata)
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
