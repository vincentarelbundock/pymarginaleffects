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
        return self.vault.get("formula", None)

    def find_variables(self):
        if "variables" in self.vault:
            return self.vault.get("variables")

        formula = self.get_formula()
        if isinstance(formula, str):
            out = fml.parse_variables(self.get_formula())
        else:
            out = None

        self.vault.update(variables=out)

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
