import numpy as np
import polars as pl
from abc import ABC, abstractmethod


class ModelAbstract(ABC):
    def __init__(self, model):
        self.model = model
        self.validate_coef()
        self.validate_modeldata()
        self.validate_response_name()
        self.validate_formula()

    def validate_coef(self):
        coef = self.get_coef()
        if not isinstance(coef, np.ndarray):
            raise ValueError("coef must be a numpy array")
        self.coef = coef

    def validate_modeldata(self):
        modeldata = self.get_modeldata()
        if not isinstance(modeldata, pl.DataFrame):
            raise ValueError("modeldata must be a Polars DataFrame")
        self.modeldata = modeldata

    def validate_response_name(self):
        response_name = self.get_response_name()
        if not isinstance(response_name, str):
            raise ValueError("response_name must be a string")
        self.response_name = response_name

    def validate_formula(self):
        formula = self.get_formula()
        if not isinstance(formula, str):
            raise ValueError("formula must be a string")
        self.formula = formula

    @abstractmethod
    def get_vcov(self):
        pass

    @abstractmethod
    def get_modeldata(self):
        pass

    @abstractmethod
    def get_response_name(self):
        pass

    # names of the variables in the original dataset, excluding interactions, intercept, etc.
    @abstractmethod
    def get_variables_names(self):
        pass

    # names of the parameters
    @abstractmethod
    def get_coef_names(self):
        pass

    @abstractmethod
    def get_predict(self):
        pass

    @abstractmethod
    def get_formula(self):
        pass
