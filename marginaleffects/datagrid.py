from functools import reduce

import polars as pl

def datagrid(
        newdata,
        FUN_numeric = lambda x: x.mean(),
        FUN_other = lambda x: x.mode()[0], # mode can return multiple values
        **kwargs):

    out = {}
    for key, value in kwargs.items():
        out[key] = pl.DataFrame({key: value})

    numtypes = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]

    for col in newdata.columns:
        # not specified manually
        if col not in out.keys():
            # numeric
            if newdata[col].dtype() in numtypes:
                out[col] = pl.DataFrame({col: FUN_numeric(newdata[col])})
            # other
            else:
                out[col] = pl.DataFrame({col: FUN_other(newdata[col])})

    out = reduce(lambda x, y: x.join(y, how="cross"), out.values())

    return out
