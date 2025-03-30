from .model_linearmodels import ModelLinearmodels
from .model_abstract import ModelAbstract
from .model_pyfixest import ModelPyfixest
from .model_statsmodels import ModelStatsmodels
from .model_sklearn import ModelSklearn


def is_linearmodels(model):
    if hasattr(model, "fit_engine") and model.fit_engine == "linearmodels":
        return True
    else:
        return False


def is_statsmodels(model):
    typename = str(type(model))
    return "statsmodels" in typename


def is_pyfixest(model):
    typename = str(type(model))
    return "pyfixest" in typename


def sanitize_model(model):
    if model is None:
        return model

    if (
        isinstance(model, ModelAbstract)
        or isinstance(model, ModelStatsmodels)
        or isinstance(model, ModelSklearn)
        or isinstance(model, ModelLinearmodels)
        or isinstance(model, ModelPyfixest)
    ):
        return model

    if is_linearmodels(model):
        return ModelLinearmodels(model)

    elif is_statsmodels(model):
        return ModelStatsmodels(model)

    elif is_pyfixest(model):
        return ModelPyfixest(model)

    raise ValueError(
        "Unknown model type. Supported modelling packages include `statsmodels` and `pyfixst`. In addition, users can call `fit_sklearn()` or `fit_linearmodels()` to fit models using the Scikit-Learn and LinearModels modelling packages."
    )
