import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from polars.testing import assert_series_equal
from tests.conftest import guerry_with_nulls, mtcars_df
from marginaleffects import *


mod = smf.ols("Literacy ~ Pop1831 * Desertion", guerry_with_nulls).fit()
mtcars_mod = smf.ols("mpg ~ hp + cyl", data=mtcars_df).fit()


def test_coefs():
    hyp_py = hypotheses(mod, hypothesis=np.array([1, -1, 0, 0]))
    hyp_r = pl.read_csv("tests/r/test_hypotheses_coefs.csv")
    assert_series_equal(hyp_r["estimate"], hyp_py["estimate"])
    assert_series_equal(hyp_r["std.error"], hyp_py["std_error"], check_names=False)


def test_hypothesis_2d_array():
    hyp_py = predictions(
        mtcars_mod, by="cyl", hypothesis=np.array([[1, 1, 2], [2, 2, 3]]).T
    )
    hyp_r = pl.read_csv("tests/r/test_hypotheses_2d_array.csv")
    assert_series_equal(hyp_r["estimate"], hyp_py["estimate"])
    assert_series_equal(hyp_r["std.error"], hyp_py["std_error"], check_names=False)


def test_comparisons():
    hyp_py = comparisons(mod, by=True, hypothesis="b0 = b1")
    hyp_r = pl.read_csv("tests/r/test_hypotheses_comparisons.csv")
    # absolute because the order of rows is different in R and Python `comparisons()` output
    assert_series_equal(hyp_r["estimate"].abs(), hyp_py["estimate"].abs())
    assert_series_equal(hyp_r["std.error"], hyp_py["std_error"], check_names=False)


def test_null_hypothesis():
    # Test with hypothesis = 0
    hyp_py_0 = hypotheses(mod, hypothesis=np.array([1, -1, 0, 0]))
    hyp_r_0 = pl.read_csv("tests/r/test_hypotheses_coefs.csv")
    assert_series_equal(hyp_r_0["estimate"], hyp_py_0["estimate"])
    assert_series_equal(hyp_r_0["std.error"], hyp_py_0["std_error"], check_names=False)

    # # Test with hypothesis = 1
    # hyp_py_1 = hypotheses(mod, hypothesis=np.array([1, -1, 0, 0]), hypothesis_null=1)
    # hyp_r_1 = pl.read_csv("tests/r/test_hypotheses_coefs_hypothesis_1.csv")
    # assert_series_equal(hyp_r_1["estimate"], hyp_py_1["estimate"])
    # assert_series_equal(hyp_r_1["std.error"], hyp_py_1["std_error"], check_names=False)


def test_hypothesis_list():
    # Hypothesis values from R
    hypothesis_values = [0.4630551, -112.8876651, -10.6664417, -5384.2708089]
    mod = smf.ols("Literacy ~ Pop1831 * Desertion", guerry_with_nulls).fit()
    hyp = hypotheses(mod, hypothesis=3)
    assert np.allclose(hyp["statistic"], hypothesis_values)
    hyp = hypotheses(mod, hypothesis=3.0)
    assert np.allclose(hyp["statistic"], hypothesis_values)


def test_coef():
    h = hypotheses(mod, hypothesis="Pop1831=Desertion")
    assert isinstance(h, pl.DataFrame)
    assert h.shape[0] == 1
    assert h["term"][0] == "Pop1831=Desertion"
