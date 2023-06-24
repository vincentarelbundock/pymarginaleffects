import polars as pl

def sanitize_newdata(fit, newdata):
    if newdata is None:
        newdata = fit.model.data.frame
    try:
        out = pl.from_pandas(newdata)
    except:
        out = newdata
    return out