import polars as pl

def sort_columns(df, by = None):
    cols = ["rowid", "term", "contrast", "estimate", "std_error", "statistic", "p_value", "conf_low", "conf_high"] + df.columns
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
    return out
