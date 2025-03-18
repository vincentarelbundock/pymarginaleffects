import narwhals as nw
import numpy as np
import polars as pl
from typing import Protocol, runtime_checkable
from pydantic import ConfigDict, validate_call
from functools import wraps
import marginaleffects.formulaic_utils as fml
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


def get_type_dictionary(formula=None, modeldata=None):
    out = dict()
    if formula is None:
        variables = modeldata.columns
    else:
        variables = fml.get_variables(formula)
    for v in variables:
        t_i = [
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        ]
        t_c = [pl.Utf8, pl.Categorical, pl.Enum]
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
    dataset: str = "thornton",
    package: str = "marginaleffects",
    docs: bool = False,
    search: str = None,
):
    """
    Download and read a dataset as a Polars DataFrame from the `marginaleffects` or from the list at https://vincentarelbundock.github.io/Rdatasets/.

    For more information, visit the website: https://marginaleffects.com/

    Or type: `help(get_dataset)`
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
        "ces_demographics": "https://marginaleffects.com/data/ces_demographics",
        "ces_survey": "https://marginaleffects.com/data/ces_survey",
        "immigration": "https://marginaleffects.com/data/immigration",
        "lottery": "https://marginaleffects.com/data/lottery",
        "military": "https://marginaleffects.com/data/military",
        "thornton": "https://marginaleffects.com/data/thornton",
        "factorial_01": "https://marginaleffects.com/data/factorial_01",
        "interaction_01": "https://marginaleffects.com/data/interaction_01",
        "interaction_02": "https://marginaleffects.com/data/interaction_02",
        "interaction_03": "https://marginaleffects.com/data/interaction_03",
        "interaction_04": "https://marginaleffects.com/data/interaction_04",
        "polynomial_01": "https://marginaleffects.com/data/polynomial_01",
        "polynomial_02": "https://marginaleffects.com/data/polynomial_02",
    }

    if package == "marginaleffects":
        assert dataset in datasets, (
            f"Dataset '{dataset}' is not available in the 'marginaleffects' package."
        )

    try:
        if dataset in datasets:
            base_url = datasets[dataset]
            df = pl.read_parquet(f"{base_url}.parquet")
            if (
                "factorial" in dataset
                or "interaction" in dataset
                or "polynomial" in dataset
            ):
                doc_url = "https://marginaleffects.com/data/model_to_meaning_simulated_data.html"
            elif dataset.startswith("ces"):
                doc_url = "https://marginaleffects.com/data/ces.html"
            else:
                doc_url = f"{base_url}.html"
        else:
            csv_url = f"https://vincentarelbundock.github.io/Rdatasets/csv/{package}/{dataset}.csv"
            doc_url = f"https://vincentarelbundock.github.io/Rdatasets/doc/{package}/{dataset}.html"
            df = pl.read_csv(csv_url)

        if docs:
            return doc_url

        return df

    except BaseException as e:
        raise ValueError(f"Error reading dataset: {e}")


def upcast(df, reference):
    numtypes = [
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    ]
    for col in df.columns:
        if col in df.columns and col in reference.columns:
            good = reference[col].dtype
            bad = df[col].dtype
            if good != bad:
                if good in numtypes and bad in numtypes:
                    idx = max(numtypes.index(good), numtypes.index(bad))
                    df = df.with_columns(pl.col(col).cast(numtypes[idx]))
                else:
                    df = df.with_columns(pl.col(col).cast(good))

    return df


get_dataset.__doc__ = """
    # `get_dataset()`


    Download and read a dataset as a Polars DataFrame from the `marginaleffects` or from the list at https://vincentarelbundock.github.io/Rdatasets/.
    Returns documentation link if `docs` is True.

    ## Parameters

    `dataset`: (str) String. Name of the dataset to download.

     - marginaleffects archive: affairs, airbnb, ces_demographics, ces_survey, immigration, lottery, military, thornton, factorial_01, interaction_01, interaction_02, interaction_03, interaction_04, polynomial_01, polynomial_02
     - Rdatasets archive: The name of a dataset listed on the Rdatasets index. See the website or the search argument.

    `package`: (str, optional) The package to download the dataset from. Default is "marginaleffects".

    `docs`: (bool, optional) If True, return the documentation URL instead of the dataset. Default is False.

    `search`: (str, optional) The string is a regular expression. Download the dataset index from Rdatasets; search the "Package", "Item", and "Title" columns; and return the matching rows.

    ## Returns
    (Union[str, pl.DataFrame])
    * A string representing the documentation URL if `docs` is True, or
        a Polars DataFrame containing the dataset if `docs` is False.

    ## Raises
    ValueError
    * If the dataset is not among the specified choices.

    ## Examples
    ```py
    get_dataset()
    get_dataset("Titanic", package="Stat2Data")
    get_dataset(search = "(?i)titanic)
    ```
    """
