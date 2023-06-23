from pytest import approx
import polars as pl
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from marginaleffects import *
from rpy2.robjects import r, pandas2ri


def rpy2_df_pandas_to_r(df):
    with (ro.default_converter + ro.pandas2ri.converter).context():
        out = ro.conversion.get_conversion().py2rpy(df)
    return out


def rpy2_df_r_to_pandas(df):
    with (ro.default_converter + ro.pandas2ri.converter).context():
        out = ro.conversion.get_conversion().rpy2py(df)
    return out


def test_interaction():
    marginaleffects = importr("marginaleffects")
    stats = importr("stats")
    df = sm.datasets.get_rdataset("Guerry", "HistData").data
    df_r = rpy2_df_pandas_to_r(df)
    mod = smf.ols("Literacy ~ Pop1831 * Desertion", df)
    fit = mod.fit()
    tmp = stats.lm("Literacy ~ Pop1831 * Desertion", data = df_r)
    cmp = marginaleffects.comparisons(tmp, variables = "Pop1831", newdata = df_r)
    known = rpy2_df_r_to_pandas(cmp)
    unknown = comparisons(fit, "Pop1831", value = 1, comparison = "difference")
    a = pl.Series(known["estimate"].to_list()).to_numpy()
    b = unknown["estimate"].to_numpy()
    assert a == approx(b)
    # a = pl.Series(known["std.error"].to_list()).to_numpy()
    # b = unknown["std_error"].to_numpy()
    # assert a == approx(b)