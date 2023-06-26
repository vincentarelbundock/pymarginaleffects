import polars as pl

def get_by(model, estimand, newdata, by = None):
    if len(estimand) == len(newdata):
        out = newdata.with_columns(pl.Series(estimand).alias("estimate"))
    else:
        out = pl.DataFrame({"estimate" : estimand})
    if by is not None:
        # maintain_order is super important
        if isinstance(by, str):
            tmp = [by] + ["estimate"]
        elif isinstance(by, list):
            tmp = by + ["estimate"]
        out = out.select(tmp).groupby(by, maintain_order=True).mean()
    return out