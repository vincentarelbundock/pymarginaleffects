import polars as pl
import statsmodels.formula.api as smf
from pytest import approx

from marginaleffects import *

dat = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/MASS/quine.csv")
mod = smf.negativebinomial("Days ~ Sex/(Age + Eth*Lrn)", data=dat.to_pandas()).fit()


def test_predictions_01():
    unknown = predictions(mod)
    known = pl.read_csv("tests/r/test_statsmodels_negativebinomial_predictions_01.csv")
    assert known["estimate"].to_numpy() == approx(
        unknown["estimate"].to_numpy(), rel=1e-3
    )


def test_comparisons_01():
    unknown = comparisons(mod, variables="Sex")
    known = pl.read_csv(
        "tests/r/test_statsmodels_negativebinomial_comparisons_01.csv"
    ).filter(pl.col("term") == "Sex")
    assert known["estimate"].to_numpy() == approx(
        unknown["estimate"].to_numpy(), rel=1e-2
    )
