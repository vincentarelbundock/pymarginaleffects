from abc import ABC, abstractmethod
from .validation import ModelValidation
from . import formulaic_utils as fml


class ModelAbstract(ModelValidation, ABC):
    def __init__(self, model):
        self.model = model
        self.formula_engine = "formulaic"
        self.vault = {}

    def get_modeldata(self):
        if "modeldata" in self.vault:
            out = self.vault.get("modeldata")
        elif hasattr(self, "data"):
            out = self.data
        else:
            out = None
        return out

    def get_vcov(self, vcov=False):
        return self.vault.get("vcov", None)

    def get_coef(self):
        return self.vault.get("coef", None)

    def get_coefnames(self):
        return self.vault.get("coefnames", None)

    def get_formula(self):
        return self.formula

    def find_variables(self):
        formula = self.get_formula()
        if "variables" in self.vault:
            out = self.vault.get("variables")
        elif isinstance(formula, str):
            out = fml.parse_variables(self.formula)
        else:
            out = None
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
