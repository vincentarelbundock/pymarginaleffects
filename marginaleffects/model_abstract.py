from abc import ABC, abstractmethod
from .utils import get_type_dictionary
from .validation import ModelValidation
from . import formulaic_utils as fml


class ModelAbstract(ModelValidation, ABC):
    def __init__(self, model):
        self.model = model
        self.formula_engine = "formulaic"
        self.validate_coef()
        self.validate_response_name()
        self.validate_formula()
        self.validate_modeldata()
        self.variables_type = get_type_dictionary(self.formula, self.data)
        self.vault = {}

    def get_vcov(self, vcov=False):
        return self.vault.get("vcov", None)

    def get_coef(self):
        return self.vault.get("coef", None)

    def get_coefnames(self):
        return None

    def find_variables(self, variables=None, newdata=None):
        out = fml.get_variables(self.formula)
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
