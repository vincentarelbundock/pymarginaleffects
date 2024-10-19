import polars as pl
import narwhals as nw


def get_by(model, estimand, newdata, by=None, wts=None):
    newdata = nw.from_native(newdata)
    estimand = estimand.with_columns(nw.col("rowid").cast(nw.dtypes.UInt32))
    # for predictions
    if (
        isinstance(by, list)
        and len(by) == 1
        and by[0] == "group"
        and "group" not in estimand.columns
    ):
        by = True

    if by is True:
        return estimand.select(["estimate"]).mean()
    elif by is False:
        return estimand

    if "group" in estimand.columns:
        by = ["group"] + by

    if "rowid" in estimand.columns and "rowid" in newdata.columns:
        out = estimand.join(newdata, on="rowid", how="left")
    else:
        out = nw.from_native(pl.DataFrame({"estimate": estimand["estimate"]}))

    by = [x for x in by if x in out.columns]
    if isinstance(by, list) and len(by) == 0:
        return out

    if wts is None:
        out = nw.from_native(
            out.to_native()
            .group_by(by, maintain_order=True)
            .agg(pl.col("estimate").mean())
        )
    else:
        out = nw.from_native(
            out.to_native()
            .group_by(by, maintain_order=True)
            .agg((pl.col("estimate") * pl.col(wts)).sum() / pl.col(wts).sum())
        )
    return out
