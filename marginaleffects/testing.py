import re
import numpy as np
from marginaleffects import *
import polars as pl
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri

def pandas_to_r(df):
    with (ro.default_converter + ro.pandas2ri.converter).context():
        out = ro.conversion.get_conversion().py2rpy(df)
    return out

def polars_to_r(df):
    return pandas_to_r(df.to_pandas())

def r_to_pandas(df):
    with (ro.default_converter + ro.pandas2ri.converter).context():
        out = ro.conversion.get_conversion().rpy2py(df)
    return out

def r_to_polars(df):
    return pl.from_pandas(r_to_pandas(df))


def compare_r_to_py(r_obj, py_obj, tolr = 1e-3, tola = 1e-3, msg = ""):
    cols = ["term", "contrast", "rowid"]
    r_obj = r_obj.sort([x for x in cols if x in r_obj.columns])
    py_obj = py_obj.sort([x for x in cols if x in py_obj.columns])
    # dont' compare other statistics because degrees of freedom don't match
    for col_py in ["estimate", "std_error"]:
        col_r = re.sub("_", ".", col_py) 
        if col_py in py_obj.columns and col_r in r_obj.columns:
            a = r_obj[col_r]
            b = py_obj[col_py]
            gap_rel = ((a - b) / a).abs().max()
            gap_abs = (a - b).abs().max()
            flag = gap_rel <= tolr or gap_abs <= tola
            assert flag, f"{msg} trel: {gap_rel}. tabs: {gap_abs}"


def rdatasets(package, dataset, r = False):
    url = f"https://vincentarelbundock.github.io/Rdatasets/csv/{package}/{dataset}.csv"
    dat_py = pl.read_csv(url)
    dat_py = dat_py.rename({"": "rownames"})
    if r is False:
        return dat_py
    else:
        dat_r = pandas_to_r(dat_py.to_pandas())
        return dat_py, dat_r