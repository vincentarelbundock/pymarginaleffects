import polars as pl

def is_numeric(column: pl.Series) -> bool:
    return column.dtype in [
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64
    ]

def is_binary(column: pl.Series) -> bool:
    if is_numeric(column):
        out = column.is_in([0, 1]).all()
    else:
        out = False
    return out

def is_character(column: pl.Series) -> bool:
    return column.dtype == pl.Utf8

def is_logical(column: pl.Series) -> bool:
    return column.dtype == pl.Boolean

def is_integer(column: pl.Series) -> bool:
    return column.dtype in [
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64
    ]

def is_other(column: pl.Series) -> bool:
    return not (is_numeric(column) or is_binary(column) or is_character(column) or is_logical(column) or is_integer(column))

