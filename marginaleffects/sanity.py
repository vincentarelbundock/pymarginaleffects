import polars as pl


def sanitize_newdata(model, newdata):
    if newdata is None:
        out = model.model.data.frame
    try:
        out = pl.from_pandas(out)
    except:
        pass
    return out


def sanitize_vcov(vcov, model):
    if isinstance(vcov, bool):
        if vcov is True:
            V = model.cov_params()
        else:
            V = None
    elif isinstance(vcov, str):
        lab = f"cov_{vcov}"
        if (hasattr(model, lab)):
            V = getattr(model, lab)
        else:
            raise ValueError(f"The model object has no {lab} attribute.")
    else:
        raise ValueError('`vcov` must be a boolean or a string like "HC3", which corresponds to an attribute of the model object such as "vcov_HC3".')
    return V

def sanitize_newdata(model, newdata):
    if newdata is None:
        newdata = model.model.data.frame
    try:
        out = pl.from_pandas(newdata)
    except:
        out = newdata
    return out