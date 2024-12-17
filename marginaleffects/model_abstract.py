import numpy as np
import polars as pl
from abc import ABC, abstractmethod
from .utils import get_type_dictionary
from . import formulaic as fml


class ModelAbstract(ABC):
    def __init__(self, model):
        if hasattr(model, "formula"):
            self.formula = model.formula
        if hasattr(model, "data"):
            self.data = model.data
        self.model = model
        self.validate_coef()
        self.validate_modeldata()
        self.validate_response_name()
        self.validate_formula()
        self.variables_type = get_type_dictionary(self.modeldata)

    def validate_coef(self):
        coef = self.get_coef()
        if not isinstance(coef, np.ndarray) and coef is not None:
            raise ValueError("coef must be a numpy array")
        self.coef = coef

    def validate_modeldata(self):
        modeldata = self.get_modeldata()
        if not isinstance(modeldata, pl.DataFrame):
            raise ValueError("modeldata must be a Polars DataFrame")
        self.modeldata = modeldata

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
        self.formula = formula

    def get_formula(self):
        if hasattr(self.model, "formula"):
            return self.model.formula
        else:
            raise ValueError("Model must have a 'formula' attribute")

    def get_modeldata(self):
        if hasattr(self.model, "data"):
            if not isinstance(self.model.data, pl.DataFrame):
                raise ValueError(
                    "The data attribute of the model must be a polars DataFrame"
                )
        return self.model.data

    def get_vcov(self, vcov=False):
        return None

    def get_coef(self):
        return None

    def find_coef(self):
        return None

    def find_variables(self, variables=None, newdata=None):
        f = self.get_formula()
        out = fml.variables(self.formula)[1:]
        return out

    def find_response(self):
        vars = self.find_variables()
        if vars is None:
            return None
        else:
            return vars[0]

    def find_predictors(self):
        vars = self.find_variables()
        if vars is None:
            return None
        else:
            return vars[1:]

    @abstractmethod
    def get_predict(self):
        pass
