import numpy as np
import polars as pl
from abc import ABC, abstractmethod
from .utils import get_type_dictionary
from . import formulaic as fml


class ModelAbstract(ABC):
    def __init__(self, model):
        self.model = model
        self.formula_engine = "formulaic"
        self.validate_coef()
        self.validate_modeldata()
        self.validate_response_name()
        self.validate_formula()
        self.variables_type = get_type_dictionary(self.data)

    def validate_coef(self):
        coef = self.get_coef()
        if not isinstance(coef, np.ndarray) and coef is not None:
            raise ValueError("coef must be a numpy array")
        self.coef = coef

    def validate_modeldata(self):
        if not isinstance(self.data, pl.DataFrame):
            raise ValueError("data attribute must be a Polars DataFrame")

    def validate_response_name(self):
        response_name = self.find_response()
        if not isinstance(response_name, str):
            raise ValueError("response_name must be a string")
        self.response_name = response_name

    def validate_formula(self):
        formula = self.formula

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

    def get_vcov(self, vcov=False):
        return None

    def get_coef(self):
        return None

    def find_coef(self):
        return None

    def find_variables(self, variables=None, newdata=None):
        out = fml.variables(self.formula)
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
