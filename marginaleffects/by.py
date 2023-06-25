import polars as pl

def get_by(model, estimand, newdata, by = None):
    if len(estimand) == len(newdata):
        out = newdata.with_columns(pl.Series(estimand).alias("estimate"))
    else:
        out = pl.DataFrame({"estimate" : estimand})
    if by is not None:
        # maintain_order is super important
        out = out.select([by, "estimate"]).groupby(by, maintain_order=True).mean()
    return out