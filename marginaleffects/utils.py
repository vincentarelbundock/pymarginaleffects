import itertools

import numpy as np
import polars as pl


def sort_columns(df, by=None, newdata=None):
    cols = [
        "rowid",
        "group",
        "term",
        "contrast",
        "estimate",
        "std_error",
        "statistic",
        "p_value",
        "s_value",
        "conf_low",
        "conf_high",
    ] + df.columns

    if by is not None:
        if isinstance(by, list):
            cols = by + cols
        else:
            cols = [by] + cols

    if isinstance(newdata, pl.DataFrame) and hasattr(newdata, "datagrid_explicit"):
        cols = newdata.datagrid_explicit + cols

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
        return None
    first = [df.slice(0, 1)] * len(uniqs)
    first = pl.concat(first)
    first = first.with_columns(uniqs.alias(colname))
    return first


def upcast(dfs: list) -> list:
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

    tmp = [df for df in dfs if type(df) is pl.DataFrame]

    if len(tmp) == 0:
        return dfs

    cols = [df.columns for df in tmp]
    cols = set(list(itertools.chain(*cols)))

    for col in cols:
        dtypes = [df[col].dtype for df in tmp if col in df.columns]
        match = [
            next((i for i, x in enumerate(numeric_types) if x == dtype), None)
            for dtype in dtypes
        ]
        match = list(set(match))
        if len(match) > 1:
            match = max(match)
            if match is not None:
                for i, v in enumerate(tmp):
                    tmp[i] = tmp[i].with_columns(pl.col(col).cast(numeric_types[match]))

    return tmp


def get_variable_type(variable, newdata):
    inttypes = [pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
    if variable not in newdata.columns:
        return None
        # raise ValueError(f"`{variable}` is not in `newdata`")
    if newdata[variable].dtype in [pl.Utf8, pl.Categorical]:
        return "character"
    elif newdata[variable].dtype == pl.Boolean:
        return "boolean"
    elif newdata[variable].dtype in inttypes:
        if newdata[variable].is_in([0, 1]).all():
            return "binary"
        else:
            return "numeric"
    elif newdata[variable].dtype in [pl.Float32, pl.Float64]:
        return "numeric"
    else:
        raise ValueError(f"Unknown type for `{variable}`: {newdata[variable].dtype}")
