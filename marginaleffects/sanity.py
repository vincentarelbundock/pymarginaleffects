import polars as pl
from .estimands import * 


def sanitize_newdata(model, newdata):
    if newdata is None:
        out = model.model.data.frame
    try:
        out = pl.from_pandas(out)
    except:
        pass
    origin = [pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float64]
    for col in out.columns:
        if out[col].dtype in origin:
            print("good")
            out = out.with_columns(pl.col(col).cast(pl.Float32).alias(col))
        else:
            print("bad")
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
    
    if "rowid" in out.columns:
        raise ValueError("The newdata has a column named 'rowid', which is not allowed.")
    else:
        out = out.with_columns(pl.Series(range(out.height)).alias("rowid"))

    return out