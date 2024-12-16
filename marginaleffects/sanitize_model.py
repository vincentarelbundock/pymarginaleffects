from .model_abstract import ModelAbstract
from .model_pyfixest import ModelPyfixest
from .model_statsmodels import ModelStatsmodels
from .model_sklearn import ModelSklearn, is_sklearn_model


def sanitize_model(model):
    if model is None:
        return model

    if isinstance(model, ModelAbstract):
        return model

    # statsmodels
    try:
        import statsmodels.base.wrapper as smw

        if isinstance(model, smw.ResultsWrapper):
            return ModelStatsmodels(model)
    except ImportError:
        pass

    # scikit-learn
    try:
        if is_sklearn_model(model):
            return ModelSklearn(model)
    except ImportError:
        pass

    # pyfixest
    try:
        import pyfixest  #  noqa

        return ModelPyfixest(model)
    except ImportError:
        pass

    raise ValueError(
        "Unknown model type. Try installing the 'statsmodels' package or file an issue at https://github.com/vincentarelbundock/pymarginaleffects."
    )
