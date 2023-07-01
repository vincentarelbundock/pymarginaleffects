import numpy as np
import polars as pl
import statsmodels.formula.api as smf
import statsmodels.api as sm
from marginaleffects import *
from pytest import approx


dat = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/geepack/dietox.csv")
mod = smf.mixedlm(formula = "Weight ~ Time * Litter", data = dat, groups=dat["Pig"]).fit()


def test_predictions_01():
    unknown = predictions(mod)
    known = pl.read_csv("tests/r/test_statsmodels_mixedlm_predictions_01.csv")
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-4)


def test_predictions_02():
    unknown = predictions(mod, by = "Cu")
    known = pl.read_csv("tests/r/test_statsmodels_mixedlm_predictions_02.csv")
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-1)


def test_comparisons_01():
    unknown = comparisons(mod)
    known = pl.read_csv("tests/r/test_statsmodels_mixedlm_comparisons_01.csv", ignore_errors=True)
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-2)


def test_comparisons_02():
    unknown = comparisons(mod, by = "Cu").sort(["term", "Cu"])
    known = pl.read_csv("tests/r/test_statsmodels_mixedlm_comparisons_02.csv").sort(["term", "Cu"])
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-2)