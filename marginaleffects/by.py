import polars as pl


def get_by(model, estimand, newdata, by=None, wts=None):

    if "group" in estimand.columns:
        by = ["group"] + by

    if "rowid" in estimand.columns and "rowid" in newdata.columns:
        out = estimand.join(newdata, on="rowid", how="left")
    else:
        out = pl.DataFrame({"estimate": estimand["estimate"]})

    if by is True:
        return out.select(["estimate"]).mean()
    elif by is False:
        return out
    
    by = [x for x in by if x in out.columns]
    if isinstance(by, list) and len(by) == 0:
        return out


    if wts is None:
        out = out.groupby(by, maintain_order=True).agg(pl.col("estimate").mean())
    else:
        out = out.groupby(by, maintain_order=True).agg(
            (pl.col("estimate") * pl.col(wts)).sum() / pl.col(wts).sum()
        )
    return out
