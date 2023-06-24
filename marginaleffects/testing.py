from marginaleffects import *
import polars as pl
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri
from pytest import approx

def pandas_to_r(df):
    with (ro.default_converter + ro.pandas2ri.converter).context():
        out = ro.conversion.get_conversion().py2rpy(df)
    return out


def r_to_pandas(df):
    with (ro.default_converter + ro.pandas2ri.converter).context():
        out = ro.conversion.get_conversion().rpy2py(df)
    return out


def r_to_polars(df):
    return pl.from_pandas(r_to_pandas(df))


def compare_r_to_py(r_obj, py_obj):
    r_df = rpy2_df_r_to_pandas(r_obj)
    for i in range(len(r_df.columns)):
        a = r_df[r_df.columns[i]].to_numpy()
        b = py_obj[r_df.columns[i]].to_numpy()
        assert a == approx(b)