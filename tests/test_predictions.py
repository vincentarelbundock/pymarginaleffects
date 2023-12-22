import polars as pl
import statsmodels.formula.api as smf

import marginaleffects
from marginaleffects import *

from .utilities import *

df = pl.read_csv(
    "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Guerry.csv",
    null_values="NA",
).drop_nulls()
df = df.with_columns(pl.Series(range(df.shape[0])).alias("row_id")).sort(
    "Region", "row_id"
)
mod_py = smf.ols("Literacy ~ Pop1831 * Desertion", df).fit()


def test_predictions():
    pre_py = predictions(mod_py)
    pre_r = pl.read_csv("tests/r/test_predictions_01.csv")
    compare_r_to_py(pre_r, pre_py)


def test_by():
    pre_py = predictions(mod_py, by="Region")
    pre_r = pl.read_csv("tests/r/test_predictions_02.csv")
    compare_r_to_py(pre_r, pre_py)


def test_by_hypothesis():
    pre_py = predictions(mod_py, by="Region")
    pre_py = predictions(mod_py, by="Region", hypothesis="b1 * b3 = b3*2")
    pre_r = pl.read_csv("tests/r/test_predictions_03.csv")
    compare_r_to_py(pre_r, pre_py)


def test_class_manipulation():
    p = predictions(mod_py)
    assert isinstance(p, pl.DataFrame)
    assert isinstance(p, marginaleffects.classes.MarginaleffectsDataFrame)
    p = p.head()
    assert isinstance(p, pl.DataFrame)
    assert isinstance(p, marginaleffects.classes.MarginaleffectsDataFrame)


def issue_38():
    p = avg_predictions(mod_py, by=True)
    assert p.shape[0] == 1
    p = avg_predictions(mod_py)
    assert p.shape[0] == 1
