import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from pytest import approx
from scipy.stats import pearsonr

dat = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv") \
    .with_columns(pl.col("cyl").cast(pl.Utf8))
mod = smf.poisson("carb ~ mpg * qsec + cyl", data = dat).fit()


def test_predictions_01():
    unknown = predictions(mod)
    known = pl.read_csv("tests/r/test_statsmodels_poisson_predictions_01.csv")
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy())


def test_predictions_02():
    unknown = predictions(mod, by = "cyl")
    known = pl.read_csv("tests/r/test_statsmodels_poisson_predictions_02.csv")
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-4)


def test_comparisons_01():
    unknown = comparisons(mod)
    known = pl.read_csv("tests/r/test_statsmodels_poisson_comparisons_01.csv")
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-2)


def test_comparisons_02():
    unknown = comparisons(mod, by = "cyl").sort(["term", "cyl"])
    known = pl.read_csv("tests/r/test_statsmodels_poisson_comparisons_02.csv").sort(["term", "cyl"])
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-2)