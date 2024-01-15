import polars as pl
from polars.testing import assert_series_equal
import numpy as np
from pyfixest.estimation import feols, fepois
from pyfixest.utils import get_data, ssc
from marginaleffects import *
import pytest

def create_test_data():

    np.random.seed(1024)
    data = pl.DataFrame({
        "X1": np.random.normal(size = 1000),
        "X2": np.random.normal(size = 1000),
        "Z1": np.random.normal(size = 1000),
        "e": np.random.normal(size = 1000),
        "f1": np.random.choice([0, 1, 2, 3, 4, 5], size = 1000, replace = True),
        "f2": np.random.choice([0, 1, 2, 3, 4, 5], size = 1000, replace = True)
    }).with_columns((pl.col("X1") * pl.col("X2") * pl.col("Z1") + pl.col("e")).alias("Y"))

    return data



def test_bare_minimum():

    data = create_test_data()

    # test 1: no fixed effects

    fit = feols("Y ~ X1 * X2 * Z1", data = data.to_pandas(), ssc = ssc(fixef_k = "none"))

    p = predictions(fit)
    assert p.shape == (1000, 15)

    p = avg_predictions(fit)
    assert_series_equal(p["estimate"], pl.Series([0.010447]),
                        check_names=False, rtol=1e-3)
    s = avg_slopes(fit)
    known = pl.Series([0.010960664156094414, -0.02592049598947146, 0.08384415120847774])
    assert_series_equal(s["estimate"], known, check_names=False, rtol=1e-4)

    c = comparisons(fit, newdata = datagrid(X1 = [2, 4], model = fit))
    known = pl.Series([0.025874865365717037, 0.02587486536571701, 0.06166037023566736, 0.14618691715896967, -0.12440034801695467, -0.2807823993887409])
    assert_series_equal(c["estimate"], known, check_names=False, rtol=1e-4)

    # test 2: fixed effects
    fit2 = feols("Y ~ X1 * X2 * Z1 | f1", data = data.to_pandas())
    p2 = predictions(fit2)
    assert p2.shape == (1000, 15)

    p2 = avg_predictions(fit2)
    assert_series_equal(p["estimate"], pl.Series([0.0104466614683 ]), check_names=False, rtol=1e-3)

    s2 = avg_slopes(fit2)
    known = pl.Series([0.0109451775035, -0.0218987575428, 0.0811949147670])
    assert_series_equal(s2["estimate"], known, check_names=False, rtol=1e-4)

    c2 = comparisons(fit2, newdata = datagrid(X1 = [2, 4], model = fit2))
    known = pl.Series([0.0258895660319, 0.0258895660319 , 0.0711836991586 , 0.1612363096034 , -0.1305279178471 , -0.2903009296021])
    assert_series_equal(c2["estimate"], known, check_names=False, rtol=1e-4)


@pytest.mark.skip(reason="predict method with newdata not yet implemented for fepois.")
def test_bare_minimum_fepois():

    data = create_test_data().to_pandas()
    data["Y"] = data["Y"].abs()

    # test 1: no fixed effects
    fit1 = fepois("Y ~ X1 * X2 * Z1", data = data)
    p1 = predictions(fit1)
    assert p1.shape == (1000, 11)

    p1 = avg_predictions(fit1)
    assert_series_equal(p1["estimate"], pl.Series([1.02079513338]), check_names=False, rtol=1e-3)

    s1 = avg_slopes(fit1)
    known = pl.Series([0.0249776642963 , -0.0301245101317, -0.0651280822962])
    assert_series_equal(s1["estimate"], known, check_names=False, rtol=1e-4)

    c1 = comparisons(fit1, newdata = datagrid(X1 = [2, 4], model = fit1))
    known = pl.Series([0.0248290488423, 0.0260243633045, -0.1686213732790, -0.3250219312182, -0.0435495908984, -0.0211557385990])
    assert_series_equal(c1["estimate"], known, check_names=False, rtol=1e-4)
