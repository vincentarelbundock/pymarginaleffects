import numpy as np
import polars as pl
import warnings
from .utils import get_type_dictionary
from . import formulaic_utils as fml


class ModelValidation:
    def validation(self):
        self.validate_coef()
        self.validate_response_name()
        self.validate_formula()
        self.validate_modeldata()
        self.variables_type = get_type_dictionary(
            self.get_formula(), self.get_modeldata()
        )

    def validate_coef(self):
        coef = self.get_coef()
        if not isinstance(coef, np.ndarray) and coef is not None:
            raise ValueError("coef must be a numpy array")
        self.coef = coef

    def validate_response_name(self):
        response_name = self.find_response()
        if not isinstance(response_name, str):
            raise ValueError("response_name must be a string")
        self.response_name = response_name

    def validate_formula(self):
        formula = self.get_formula()

        if not isinstance(formula, str):
            raise ValueError("formula must be a string")

        if "~" not in formula:
            raise ValueError(
                "Model formula must contain '~' to separate dependent and independent variables"
            )

        if "scale(" in formula or "center(" in formula:
            raise ValueError(
                "The formula cannot include scale( or center(. Please center your variables before fitting the model."
            )

        # TODO: deduplicate once we only use the vault
        self.vault.update(formula=formula)
        self.formula = formula

    def validate_modeldata(self):
        modeldata = self.get_modeldata()

        if not isinstance(modeldata, pl.DataFrame):
            raise ValueError("data attribute must be a Polars DataFrame")

        # there can be no missing values in the formula variables
        original_row_count = modeldata.shape[0]
        modeldata = fml.listwise_deletion(self.get_formula(), modeldata)
        if modeldata.shape[0] != original_row_count:
            warnings.warn("Dropping rows with missing observations.", UserWarning)

        # categorical variables must be encoded as such
        catvars = fml.parse_variables_categorical(self.get_formula())
        for c in catvars:
            if modeldata[c].dtype not in [pl.Enum, pl.Categorical]:
                if modeldata[c].dtype.is_numeric():
                    msg = f"Variable {c} is numeric. It should be String, Categorical, or Enum."
                    raise ValueError(msg)
                catvals = modeldata[c].unique().sort().drop_nulls()
                modeldata = modeldata.with_columns(pl.col(c).cast(pl.Categorical))
                modeldata = modeldata.with_columns(pl.col(c).cast(pl.Enum(catvals)))

        for c in modeldata.columns:
            if modeldata[c].dtype in [pl.Utf8, pl.String]:
                catvals = modeldata[c].unique().sort().drop_nulls()
                modeldata = modeldata.with_columns(pl.col(c).cast(pl.Enum(catvals)))
            elif modeldata[c].dtype in [pl.Categorical]:
                catvals = modeldata[c].cat.get_categories().drop_nulls()
                modeldata = modeldata.with_columns(pl.col(c).cast(pl.Enum(catvals)))

        self.vault.update(modeldata=modeldata)
