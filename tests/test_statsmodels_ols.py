import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from pytest import approx

dat = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv")
dat = dat.with_columns(pl.col("cyl").cast(pl.Utf8))
mod = smf.ols("mpg ~ qsec * wt + cyl", data = dat).fit()


def test_predictions_01():
    unknown = predictions(mod)
    known = pl.read_csv("tests/r/test_statsmodels_ols_predictions_01.csv")
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-4)
    assert known["std.error"].to_numpy() == approx(unknown["std_error"].to_numpy(), rel = 1e-4)


def test_predictions_02():
    unknown = predictions(mod, by = "carb")
    known = pl.read_csv("tests/r/test_statsmodels_ols_predictions_02.csv")
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-4)
    assert known["std.error"].to_numpy() == approx(unknown["std_error"].to_numpy(), rel = 1e-4)


def test_comparisons_01():
    unknown = comparisons(mod)
    known = pl.read_csv("tests/r/test_statsmodels_ols_comparisons_01.csv")
    unknown = unknown.sort(["term", "contrast", "rowid"])
    known = known.sort(["term", "contrast", "rowid"])
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-4)
    assert known["std.error"].to_numpy() == approx(unknown["std_error"].to_numpy(), rel = 1e-4)


def test_comparisons_02():
    unknown = comparisons(mod, by = "carb").sort(["term", "carb"])
    known = pl.read_csv("tests/r/test_statsmodels_ols_comparisons_02.csv").sort(["term", "carb"])
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-4)
    assert known["std.error"].to_numpy() == approx(unknown["std_error"].to_numpy(), rel = 1e-4)


def test_slopes_01():
    unknown = slopes(mod)
    known = pl.read_csv("tests/r/test_statsmodels_ols_slopes_01.csv")
    unknown = unknown.sort(["term", "contrast", "rowid"])
    known = known.sort(["term", "contrast", "rowid"])
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 2e-2)
    assert known["std.error"].to_numpy() == approx(unknown["std_error"].to_numpy(), rel = 2e-2)


def test_slopes_02():
    unknown = slopes(mod, by = "carb").sort(["term", "carb"])
    known = pl.read_csv("tests/r/test_statsmodels_ols_slopes_02.csv").sort(["term", "carb"])
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-3)
    assert known["std.error"].to_numpy() == approx(unknown["std_error"].to_numpy(), rel = 1e-4)
