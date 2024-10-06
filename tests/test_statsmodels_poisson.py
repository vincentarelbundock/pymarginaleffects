import polars as pl
import statsmodels.formula.api as smf
from polars.testing import assert_series_equal

from marginaleffects import comparisons, predictions
from .conftest import mtcars_df

dat = mtcars_df.with_columns(pl.col("cyl").cast(pl.Utf8))
mod = smf.poisson("carb ~ mpg * qsec + cyl", data=dat.to_pandas()).fit()


def test_predictions_01():
    unknown = predictions(mod)
    known = pl.read_csv("tests/r/test_statsmodels_poisson_predictions_01.csv")
    assert_series_equal(known["estimate"], unknown["estimate"], rtol=1e-4)


def test_predictions_02():
    unknown = predictions(mod, by="cyl")
    known = pl.read_csv("tests/r/test_statsmodels_poisson_predictions_02.csv")
    assert_series_equal(known["estimate"], unknown["estimate"], rtol=1e-4)


def test_comparisons_01():
    unknown = comparisons(mod).sort(["term", "contrast"])
    known = pl.read_csv("tests/r/test_statsmodels_poisson_comparisons_01.csv").sort(
        ["term", "contrast"]
    )
    assert_series_equal(known["estimate"], unknown["estimate"], rtol=1e-4)


def test_comparisons_02():
    unknown = comparisons(mod, by="cyl").sort(["term", "contrast", "cyl"])
    known = pl.read_csv("tests/r/test_statsmodels_poisson_comparisons_02.csv").sort(
        ["term", "contrast", "cyl"]
    )
    assert_series_equal(known["estimate"], unknown["estimate"], rtol=1e-4)
