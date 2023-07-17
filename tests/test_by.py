import re
from pytest import approx
import polars as pl
from marginaleffects import *
from .utilities import *
import statsmodels.formula.api as smf

Guerry = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Guerry.csv", null_values = "NA").drop_nulls()
mod_py = smf.ols("Literacy ~ Pop1831 * Desertion", Guerry).fit()


def test_predictions_by_string():
    cmp_py = predictions(mod_py, by = "Region")
    cmp_r = pl.read_csv("tests/r/test_by_01.csv")
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
    pre_r = pl.read_csv("tests/r/test_by_04.csv")
    compare_r_to_py(pre_r, pre_py)