import polars as pl
import numpy as np

def sort_columns(df, by = None):
    cols = ["rowid", "group", "term", "contrast", "estimate", "std_error", "statistic", "p_value", "s_value", "conf_low", "conf_high"] + df.columns
    if by is not None:
        if isinstance(by, list):
            cols = by + cols
        else:
            cols = [by] + cols
    cols = [x for x in cols if x in df.columns]
    cols_unique = []
    for item in cols:
        if item not in cols_unique:
            cols_unique.append(item)
    out = df.select(cols_unique)
    if "marginaleffects_comparison" in out.columns:
        out = out.drop("marginaleffects_comparison")
    return out


def pad_array(arr, n):
    if len(arr) == 1:
        out = np.repeat(arr[0], n)
    elif len(arr) < n:
        out = np.concatenate([np.repeat(arr[0], n - len(arr)), arr])
    else:
        out = arr
    return pl.Series(out)

def get_pad(df, colname, uniqs):
    if uniqs is None:
        return(None)
    first = [df.slice(0, 1)] * len(uniqs)
    first = pl.concat(first)
    first = first.with_columns(uniqs.alias(colname))
    return first


def convert_int_columns_to_float32(dfs: list) -> list:
    numeric_types = [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    ]

    converted_dfs = []
    for df in dfs:
        new_columns = []
        if df is not None:
            for col in df:
                if col.dtype in numeric_types:
                    new_columns.append(col.cast(pl.Float32).alias(col.name))
                else:
                    new_columns.append(col)
            converted_df = df.with_columns(new_columns)
            converted_dfs.append(converted_df)
    return converted_dfs

