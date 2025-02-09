from .model_linearmodels import ModelLinearmodels
from .model_abstract import ModelAbstract
from .model_pyfixest import ModelPyfixest
from .model_statsmodels import ModelStatsmodels
from .model_sklearn import ModelSklearn


def is_linearmodels(model):
    if hasattr(model, "fit_engine") and model.fit_engine == "statsmodels":
        return True
    else:
        return False


def is_sklearn(model):
    if hasattr(model, "fit_engine") and model.fit_engine == "statsmodels":
        return True
    try:
        from sklearn.base import BaseEstimator

        return isinstance(model, BaseEstimator) or model.__module__.startswith(
            "sklearn"
        )
    except (AttributeError, ImportError):
        return False


def is_statsmodels(model):
    if hasattr(model, "fit_engine") and model.fit_engine == "statsmodels":
        return True
    try:
        import statsmodels.base.wrapper as smw

        if isinstance(model, smw.ResultsWrapper):
            return True
        else:
            return False
    except ImportError:
        return False


def sanitize_model(model):
    if model is None:
        return model

    if (
        isinstance(model, ModelAbstract)
        or isinstance(model, ModelStatsmodels)
        or isinstance(model, ModelSklearn)
    ):
        return model

    if is_linearmodels(model):
        return ModelLinearmodels(model)

    elif is_statsmodels(model):
        return ModelStatsmodels(model)

    elif is_sklearn(model):
        return ModelSklearn(model)

    try:
        from linearmodels.panel.results import PanelResults

        if isinstance(model, PanelResults):
            return ModelLinearmodels(model)
    except ImportError:
        pass

    try:
        import pyfixest  #  noqa

        return ModelPyfixest(model)
    except ImportError:
        pass

    raise ValueError(
        "Unknown model type. Try installing the 'statsmodels' package or file an issue at https://github.com/vincentarelbundock/pymarginaleffects."
    )
