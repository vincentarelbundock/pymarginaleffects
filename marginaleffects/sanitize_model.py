from .model_abstract import ModelAbstract
from .model_pyfixest import ModelPyfixest
from .model_statsmodels import ModelStatsmodels
from .model_scikit import ModelScikitLogisticRegression


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
        from sklearn.linear_model import LogisticRegression
        if isinstance(model, LogisticRegression):
            return ModelScikitLogisticRegression(model)
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
