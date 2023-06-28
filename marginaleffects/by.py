import polars as pl

def get_by(model, estimand, newdata, by = None):
    if "rowid" in estimand.columns and "rowid" in newdata.columns:
        out = estimand.join(newdata, on = "rowid", how = "left")
    else:
        out = pl.DataFrame({"estimate" : estimand["estimate"]})
    if by is True:
        return out.select(["estimate"]).mean()
    elif by is False:
        return out
    elif not isinstance(by, list) and not isinstance(by, str):
         raise ValueError("by must be True, False, str, or list")
    out = out \
        .groupby(by, maintain_order=True) \
        .agg(pl.col("estimate").mean())
    return out