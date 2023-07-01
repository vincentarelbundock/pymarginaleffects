import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from pytest import approx
from scipy.stats import pearsonr

dat = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/iris.csv")
dat = dat.rename({"Sepal.Length": "Sepal_Length", "Sepal.Width": "Sepal_Width", "Petal.Length": "Petal_Length", "Petal.Width": "Petal_Width"})
mod = smf.wls("Sepal_Width ~ Petal_Length * Petal_Width", data = dat, weights = dat["Sepal_Length"]).fit()


def test_predictions_01():
    unknown = predictions(mod)
    known = pl.read_csv("tests/r/test_statsmodels_wls_predictions_01.csv", ignore_errors = True)
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-4)


def test_predictions_02():
    unknown = predictions(mod, by = "Species")
    known = pl.read_csv("tests/r/test_statsmodels_wls_predictions_02.csv")
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-4)


def test_comparisons_01():
    unknown = comparisons(mod).sort("term")
    known = pl.read_csv("tests/r/test_statsmodels_wls_comparisons_01.csv").sort("term")
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-2)


def test_comparisons_02():
    unknown = comparisons(mod, by = "Species").sort(["term", "Species"])
    known = pl.read_csv("tests/r/test_statsmodels_wls_comparisons_02.csv").sort(["term", "Species"])
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-2)