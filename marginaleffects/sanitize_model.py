from .model_abstract import ModelAbstract, ModelStatsmodels

def sanitize_model(model):
    # TODO: other than statsmodels
    if model is None:
        return model

    if not isinstance(model, ModelAbstract):
        model = ModelStatsmodels(model)
    return model