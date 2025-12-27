import polars as pl
import numpy as np


def get_by(model, estimand, newdata, by=None, wts=None):
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
        out = pl.DataFrame({"estimate": estimand["estimate"]})

    by = [x for x in by if x in out.columns]
    by = np.unique(by)

    if isinstance(by, list) and len(by) == 0:
        return out

    if wts is None:
        out = out.group_by(by, maintain_order=True).agg(pl.col("estimate").mean())
    else:
        out = out.group_by(by, maintain_order=True).agg(
            (pl.col("estimate") * pl.col(wts)).sum() / pl.col(wts).sum()
        )

    # Sort by 'by' columns ONLY if they are Enum type to ensure consistent categorical ordering
    # For Enum columns, sort() respects the enum order (not lexical order)
    # For other types (strings, numbers), maintain the group_by order to preserve existing behavior
    if isinstance(by, str):
        by_cols = [by]
    else:
        by_cols = list(by)

    should_sort = any(
        out[col].dtype == pl.Enum for col in by_cols if col in out.columns
    )
    if should_sort:
        out = out.sort(by)

    return out
