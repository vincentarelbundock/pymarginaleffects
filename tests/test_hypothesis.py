import numpy as np
import statsmodels.formula.api as smf
from marginaleffects import *
import polars as pl
from polars.testing import assert_series_equal

mtcars = get_dataset("mtcars", "datasets")
mod = smf.ols("mpg ~ hp + cyl", data=mtcars.to_pandas()).fit()
p = predictions(mod, by="cyl")["estimate"]


def test_predictions_reference():
    q = predictions(mod, by="cyl", hypothesis="difference~reference")["estimate"]
    assert np.isclose(p[1] - p[0], q[0])
    assert np.isclose(p[2] - p[0], q[1])
    q = predictions(mod, by="cyl", hypothesis="ratio~reference")["estimate"]
    assert np.isclose(p[1] / p[0], q[0])
    assert np.isclose(p[2] / p[0], q[1])
    q = predictions(mod, by="cyl", hypothesis="difference~revreference")["estimate"]
    assert np.isclose(p[0] - p[1], q[0])
    assert np.isclose(p[0] - p[2], q[1])
    q = predictions(mod, by="cyl", hypothesis="ratio~revreference")["estimate"]
    assert np.isclose(p[0] / p[1], q[0])
    assert np.isclose(p[0] / p[2], q[1])


def test_predictions_sequential():
    q = predictions(mod, by="cyl", hypothesis="difference~sequential")["estimate"]
    assert np.isclose(p[1] - p[0], q[0])
    assert np.isclose(p[2] - p[1], q[1])
    q = predictions(mod, by="cyl", hypothesis="ratio~sequential")["estimate"]
    assert np.isclose(p[1] / p[0], q[0])
    assert np.isclose(p[2] / p[1], q[1])
    q = predictions(mod, by="cyl", hypothesis="difference~sequential")["estimate"]
    assert np.isclose(p[1] - p[0], q[0])
    assert np.isclose(p[2] - p[1], q[1])
    q = predictions(mod, by="cyl", hypothesis="ratio~sequential")["estimate"]
    assert np.isclose(p[1] / p[0], q[0])
    assert np.isclose(p[2] / p[1], q[1])


def test_predictions_pairwise():
    p = predictions(mod, by="cyl")["estimate"]
    q = predictions(mod, by="cyl", hypothesis="ratio~pairwise")["estimate"]
    assert np.isclose(p[1] / p[0], q[0])
    assert np.isclose(p[2] / p[0], q[1])
    assert np.isclose(p[2] / p[1], q[2])
    p = predictions(mod, by="cyl")["estimate"]
    q = predictions(mod, by="cyl", hypothesis="difference~pairwise")["estimate"]
    assert np.isclose(p[1] - p[0], q[0])
    assert np.isclose(p[2] - p[0], q[1])
    assert np.isclose(p[2] - p[1], q[2])
    p = predictions(mod, by="cyl")["estimate"]
    q = predictions(mod, by="cyl", hypothesis="ratio~revpairwise")["estimate"]
    assert np.isclose(p[0] / p[1], q[0])
    assert np.isclose(p[0] / p[2], q[1])
    assert np.isclose(p[1] / p[2], q[2])
    p = predictions(mod, by="cyl")["estimate"]
    q = predictions(mod, by="cyl", hypothesis="difference~revpairwise")["estimate"]
    assert np.isclose(p[0] - p[1], q[0])
    assert np.isclose(p[0] - p[2], q[1])
    assert np.isclose(p[1] - p[2], q[2])


def test_comparisons_by():
    mtcars = (
        get_dataset("mtcars", "datasets")
        .sort("cyl")
        .with_columns(pl.col("cyl").cast(pl.String))
        .to_pandas()
    )
    mod = smf.ols("mpg ~ hp * C(cyl)", data=mtcars).fit()
    q = avg_comparisons(mod, hypothesis="ratio~sequential")
    assert q.shape[0] == 2

    q = avg_comparisons(
        mod, variables="hp", by="cyl", hypothesis="difference~revpairwise"
    )
    assert q["estimate"][0] < 0
    assert q["estimate"][1] < 0
    assert q["estimate"][2] > 0
    assert q.shape[0] == 3


def test_hypothesis_by_01():
    dat = mtcars.with_columns(pl.col("cyl").cast(pl.String))
    mod = smf.ols("mpg ~ hp * C(cyl) * am", data=dat.to_pandas()).fit()
    p = avg_predictions(mod, by=["am", "cyl"], hypothesis="~ reference | am")
    r_b = pl.Series([-3.775, -7.85, -7.50833333333336, -12.675])
    r_se = pl.Series(
        [2.19340004101813, 1.85376132615083, 1.94424126381778, 2.27038464937531]
    )
    assert_series_equal(p["estimate"], r_b, check_names=False)
    assert_series_equal(p["std_error"], r_se, check_names=False)


def numpy_formula():
    dat = get_dataset("thornton")
    dat.head(6)
    mod = ols("outcome ~ agecat - 1", data=dat.to_pandas()).fit()
    h = hypotheses(mod, hypothesis="b1**2 * np.exp(b0) = 0")
    assert h.height == 1
