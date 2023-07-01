from functools import reduce

import polars as pl


def datagrid(newdata, **kwargs):
    out = {}
    for key, value in kwargs.items():
        out[key] = pl.DataFrame({key: value})

    for col in newdata.columns:
        if col not in out.keys():
            if newdata[col].dtype() in [pl.Float64, pl.Float32]:
                out[col] = newdata.select(col).mean()
            else:
                # .mode() can return multiple values
                out[col] = pl.DataFrame({col: newdata[col].mode()[0]})

    out = reduce(lambda x, y: x.join(y, how="cross"), out.values())

    return out
