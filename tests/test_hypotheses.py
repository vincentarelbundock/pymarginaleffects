import statsmodels.formula.api as smf
import numpy as np
from marginaleffects import *
import polars as pl
from polars.testing import assert_series_equal

dat = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Guerry.csv")

mod = smf.ols("Literacy ~ Pop1831 * Desertion", dat).fit()

def test_coefs():
    hyp_py = hypotheses(mod, hypothesis = np.array([1, -1, 0, 0]))
    hyp_r = pl.read_csv("tests/r/test_hypotheses_coefs.csv")
    assert_series_equal(hyp_r["estimate"], hyp_py["estimate"])
    assert_series_equal(hyp_r["std.error"], hyp_py["std_error"], check_names = False)


def test_comparisons():
    hyp_py = comparisons(mod, by = True, hypothesis = "b1 = b2")
    hyp_r = pl.read_csv("tests/r/test_hypotheses_comparisons.csv")
    # absolute because the order of rows is different in R and Python `comparisons()` output
    assert_series_equal(hyp_r["estimate"].abs(), hyp_py["estimate"].abs())
    assert_series_equal(hyp_r["std.error"], hyp_py["std_error"], check_names = False)
