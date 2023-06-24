import polars as pl

def sanitize_vcov(vcov, fit):
    if isinstance(vcov, bool):
        if vcov is True:
            V = fit.cov_params()
        else:
            V = None
    elif isinstance(vcov, str):
        lab = f"cov_{vcov}"
        if (hasattr(fit, lab)):
            V = getattr(fit, lab)
        else:
            raise ValueError(f"The fit object has no {lab} attribute.")
    else:
        raise ValueError('`vcov` must be a boolean or a string like "HC3", which corresponds to an attribute of the fit object such as "vcov_HC3".')
    return V

def sanitize_newdata(fit, newdata):
    if newdata is None:
        newdata = fit.model.data.frame
    try:
        out = pl.from_pandas(newdata)
    except:
        out = newdata
    return out