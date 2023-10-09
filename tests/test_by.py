import pytest
import polars as pl
from pytest import approx
import polars as pl
from marginaleffects import *
from .utilities import *
import statsmodels.formula.api as smf

Guerry = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Guerry.csv", null_values = "NA").drop_nulls()
mod_py = smf.ols("Literacy ~ Pop1831 * Desertion", Guerry).fit()


def test_predictions_by_string():
    cmp_py = predictions(mod_py, by = "Region")
    cmp_r = pl.read_csv("tests/r/test_by_01.csv").sort("Region")
    compare_r_to_py(cmp_r, cmp_py)


def test_comparisons_by_true():
    cmp_py = comparisons(mod_py, by = True)
    cmp_r = pl.read_csv("tests/r/test_by_02.csv")
    compare_r_to_py(cmp_r, cmp_py)


def test_comparisons_by_false():
    cmp_py = comparisons(mod_py, by = False)
    cmp_r = pl.read_csv("tests/r/test_by_03.csv")
    compare_r_to_py(cmp_r, cmp_py)


def test_predictions_by_wts():
    pre_py = predictions(mod_py, by = "Region", wts = "Donations")
    pre_r = pl.read_csv("tests/r/test_by_04.csv").sort("Region")
    compare_r_to_py(pre_r, pre_py)


########### snapshot tests don't work

# @pytest.fixture
# def predictions_fixture():
#     df = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv") \
#         .with_columns(pl.col("cyl").cast(pl.Utf8))
#     mod = smf.ols("mpg ~ hp * qsec + cyl", df).fit()
#     p = predictions(mod, by = "cyl")
#     return p

# def test_predictions_snapshot_order(predictions_fixture, snapshot):
#     snapshot.assert_match(predictions_fixture.to_csv(index=False))