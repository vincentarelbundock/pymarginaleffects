import polars as pl

def get_by(fit, estimand, df = None, by = None):
    if df is None:
        df = pl.from_pandas(fit.model.data.frame)
    if len(estimand) == len(df):
        out = df.with_columns(pl.Series(estimand).alias("estimate"))
    else:
        out = pl.DataFrame({"estimate" : estimand})
    if by is not None:
        # maintain_order is super important
        out = out.select([by, "estimate"]).groupby(by, maintain_order=True).mean()
    return out