import itertools
import narwhals as nw
import numpy as np
import polars as pl
from typing import Protocol, runtime_checkable
from pydantic import ConfigDict, validate_call
from functools import wraps
# from narwhals.typing import IntoFrame


@runtime_checkable
class ArrowStreamExportable(Protocol):
    def __arrow_c_stream__(self, requested_schema: object | None = None) -> object: ...


def ingest(df: ArrowStreamExportable):
    """
    Convert any DataFrame to a Polars DataFrame.

    Parameters
    ----------
    df : ArrowStreamExportable
        The DataFrame to convert.

    Returns
    -------
    pl.DataFrame

    Notes
    -----

    If the original DataFrame was a pandas DataFrame, the index will
    be reset to ensure compatibility with linearmodels.
    """

    try:
        import pandas as pd

        if isinstance(df, pd.DataFrame):
            df = df.reset_index()
    except ImportError:
        raise ValueError("Please install pandas to handle Pandas DataFrame as input.")

    return nw.from_arrow(df, native_namespace=pl).to_native()


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
        pl.Boolean,
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


def get_type_dictionary(modeldata):
    out = dict()
    for v in modeldata.columns:
        t_i = [pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
        t_c = [pl.Utf8, pl.Categorical]
        t_n = [pl.Float32, pl.Float64]
        t_b = [pl.Boolean]
        if modeldata[v].dtype in t_i:
            if modeldata[v].is_in([0, 1]).all():
                out[v] = "boolean"
            else:
                out[v] = "integer"
        elif modeldata[v].dtype in t_c:
            out[v] = "character"
        elif modeldata[v].dtype in t_b:
            out[v] = "boolean"
        elif modeldata[v].dtype in t_n:
            out[v] = "numeric"
        else:
            out[v] = "unknown"
    return out


def validate_types(func):
    """Decorator that validates types with arbitrary types allowed"""
    validator = validate_call(config=ConfigDict(arbitrary_types_allowed=True))(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        return validator(*args, **kwargs)

    return wrapper


def get_dataset(
    dataset: str = "ArgentinaCPI",
    package: str = "AER",
    docs: bool = False,
    search: str = None,
):
    """
    Download and read a dataset as a Polars DataFrame from the `marginaleffects` or from the list at https://vincentarelbundock.github.io/Rdatasets/.
    Returns documentation link if `docs` is True.

    Parameters
    ----------
    dataset : str
        The dataset to download. One of "affairs", "airbnb", "immigration", "military", "thornton" or Rdatasets
    package : str, optional
        The package to download the dataset from. Default is "marginaleffects".
    docs : bool, optional
        If True, return the documentation URL instead of the dataset. Default is False.
    search: str, optional
        The string is a regular expresion. Download the dataset index from Rdatasets; search the "Package", "Item", and "Title" columns; and return the matching rows.

    Returns
    -------
    Union[str, pl.DataFrame]
        A string representing the documentation URL if `docs` is True, or
        a Polars DataFrame containing the dataset if `docs` is False.

    Raises
    ------
    ValueError
        If the dataset is not among the specified choices.
    """
    if search:
        try:
            index = pl.read_csv(
                "https://vincentarelbundock.github.io/Rdatasets/datasets.csv"
            )
            index = index.filter(
                index["Package"].str.contains(search)
                | index["Item"].str.contains(search)
                | index["Title"].str.contains(search)
            )
            return index.select(["Package", "Item", "Title", "Rows", "Cols", "CSV"])
        except BaseException as e:
            raise ValueError(f"Error searching dataset: {e}")

    datasets = {
        "affairs": "https://marginaleffects.com/data/affairs",
        "airbnb": "https://marginaleffects.com/data/airbnb",
        "immigration": "https://marginaleffects.com/data/immigration",
        "military": "https://marginaleffects.com/data/military",
        "thornton": "https://marginaleffects.com/data/thornton",
    }

    try:
        if dataset in datasets:
            base_url = datasets[dataset]
            df = pl.read_parquet(f"{base_url}.parquet")
            doc_url = (
                "https://github.com/vincentarelbundock/marginaleffects/issues/1368"
            )
        else:
            csv_url = f"https://vincentarelbundock.github.io/Rdatasets/csv/{package}/{dataset}.csv"
            doc_url = f"https://vincentarelbundock.github.io/Rdatasets/doc/{package}/{dataset}.html"
            df = pl.read_csv(csv_url)

        if docs:
            return doc_url

        return df

    except BaseException as e:
        raise ValueError(f"Error reading dataset: {e}")
