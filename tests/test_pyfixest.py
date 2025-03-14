from pyfixest.estimation import feols, fepois
from pyfixest.utils import ssc
import polars as pl
from polars.testing import assert_series_equal
import numpy as np
from marginaleffects import *
import pytest

rtol = 1e-4


def create_test_data():
    np.random.seed(1024)
    data = pl.DataFrame(
        {
            "X1": np.random.normal(size=1000),
            "X2": np.random.normal(size=1000),
            "Z1": np.random.normal(size=1000),
            "e": np.random.normal(size=1000),
            "f1": np.random.choice([0, 1, 2, 3, 4, 5], size=1000, replace=True),
            "f2": np.random.choice([0, 1, 2, 3, 4, 5], size=1000, replace=True),
        }
    ).with_columns(
        (pl.col("X1") * pl.col("X2") * pl.col("Z1") + pl.col("e")).alias("Y"),
        pl.col("f1").cast(pl.Utf8),
        pl.col("f2").cast(pl.Utf8),
    )
    return data


# @pytest.mark.skipif(sys.version_info > (3, 11), reason="Requires Python 3.11 or lower")
def test_bare_minimum():
    data = create_test_data()

    # test 1: no fixed effects
    fit = feols("Y ~ X1 * X2 * Z1", data=data, ssc=ssc(fixef_k="none"))

    p = predictions(fit)
    assert p.shape == (1000, 16)

    p = avg_predictions(fit)
    assert_series_equal(
        p["estimate"], pl.Series([0.010447]), check_names=False, rtol=rtol
    )
    s = avg_slopes(fit)
    known = pl.Series([0.010960664156094414, -0.02592049598947146, 0.08384415120847774])
    assert_series_equal(s["estimate"], known, check_names=False, rtol=rtol)

    c = comparisons(fit, newdata=datagrid(X1=[2, 4], model=fit))
    known = pl.Series(
        [
            0.025874865365717037,
            0.02587486536571701,
            0.06166037023566736,
            0.14618691715896967,
            -0.12440034801695467,
            -0.2807823993887409,
        ]
    )
    assert_series_equal(c["estimate"], known, check_names=False, rtol=rtol)

    # test 2: fixed effects
    fit2 = feols("Y ~ X1 * X2 * Z1 | f1", data=data)
    p2 = predictions(fit2)
    assert p2.shape == (1000, 16)

    p2 = avg_predictions(fit2)
    assert_series_equal(
        p["estimate"], pl.Series([0.0104466614683]), check_names=False, rtol=rtol
    )

    s2 = avg_slopes(fit2)
    known = pl.Series([0.0109451775035, -0.0218987575428, 0.0811949147670])
    assert_series_equal(s2["estimate"], known, check_names=False, rtol=rtol)

    c2 = comparisons(fit2, newdata=datagrid(X1=[2, 4], model=fit2))
    known = pl.Series(
        [
            0.0258895660319,
            0.0258895660319,
            0.0711836991586,
            0.1612363096034,
            -0.1305279178471,
            -0.2903009296021,
        ]
    )
    assert_series_equal(c2["estimate"], known, check_names=False, rtol=rtol)

    # dontrun as bug in pyfixest with ^ interaction for fixed effects and predict()
    # test 3: special syntax - interacted fixed effects
    # fit3 = feols("Y ~ X1 * X2 * Z1 | f1^f2", data=data)
    # p3 = predictions(fit3)
    # assert p3.shape == (1000, 16)
    #
    # p3 = avg_predictions(fit3)
    # assert_series_equal(
    #     p3["estimate"], pl.Series([0.01044666147]), check_names=False, rtol=rtol
    # )
    #
    # s3 = avg_slopes(fit3)
    # known = pl.Series([-0.002662644695, -0.012290790756, 0.090667738344])
    # assert_series_equal(s3["estimate"], known, check_names=False, rtol=rtol)
    #
    # c3 = comparisons(fit3, newdata=datagrid(X1=[2, 4], model=fit3))
    # known = pl.Series(
    #     [
    #         0.01222839154,
    #         0.01222839154,
    #         0.06484658716,
    #         0.13887504383,
    #         -0.11390247943,
    #         -0.26667151028,
    #     ]
    # )
    # assert_series_equal(c3["estimate"], known, check_names=False, rtol=rtol)
    #


@pytest.mark.skip(reason="predict method with newdata not yet implemented for fepois.")
def test_bare_minimum_fepois():
    data = create_test_data()
    data = data.with_columns(pl.col("Y").abs().round())

    # test 1: no fixed effects
    fit1 = fepois("Y ~ X1 * X2 * Z1", data=data)
    p1 = predictions(fit1)
    assert p1.shape == (1000, 11)

    p1 = avg_predictions(fit1)
    assert_series_equal(
        p1["estimate"], pl.Series([1.02079513338]), check_names=False, rtol=rtol
    )

    s1 = avg_slopes(fit1)
    known = pl.Series([0.0249776642963, -0.0301245101317, -0.0651280822962])
    assert_series_equal(s1["estimate"], known, check_names=False, rtol=rtol)

    c1 = comparisons(fit1, newdata=datagrid(X1=[2, 4], model=fit1))
    known = pl.Series(
        [
            0.0248290488423,
            0.0260243633045,
            -0.1686213732790,
            -0.3250219312182,
            -0.0435495908984,
            -0.0211557385990,
        ]
    )
    assert_series_equal(c1["estimate"], known, check_names=False, rtol=rtol)
