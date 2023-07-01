import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from pytest import approx

dat = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/iris.csv")
dat = dat.rename({"Sepal.Length": "Sepal_Length", "Sepal.Width": "Sepal_Width", "Petal.Length": "Petal_Length", "Petal.Width": "Petal_Width"})
dat = dat.drop_nulls("Species")
mod = smf.quantreg("Sepal_Length ~ Sepal_Width * Petal_Length + Species", data = dat).fit(.25)


def test_predictions_01():
    unknown = predictions(mod)
    known = pl.read_csv("tests/r/test_statsmodels_quantreg_predictions_01.csv")
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-2)


def test_predictions_02():
    unknown = predictions(mod, by = "Species")
    known = pl.read_csv("tests/r/test_statsmodels_quantreg_predictions_02.csv")
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-2)


def test_comparisons_01():
    unknown = comparisons(mod) \
        .rename({"estimate": "estimate_unknown"}) \
        .select(["rowid", "term", "contrast", "Species", "estimate_unknown"]) \
        .with_columns(pl.col("term").str.replace("_", "."),
                      pl.col("rowid").cast(pl.Int64))
    known = pl.read_csv("tests/r/test_statsmodels_quantreg_comparisons_01.csv")
    tmp = known.join(unknown, on = ["rowid", "term", "contrast", "Species"], how = "left")
    assert tmp["estimate"].to_numpy() == approx(tmp["estimate_unknown"].to_numpy(), rel = 2e-1)


def test_comparisons_02():
    unknown = comparisons(mod, by = "Species").sort(["term", "Species"])
    known = pl.read_csv("tests/r/test_statsmodels_quantreg_comparisons_02.csv").sort(["term", "Species"])
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-4)
