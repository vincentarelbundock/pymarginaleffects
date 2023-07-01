import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from pytest import approx

dat = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/palmerpenguins/penguins.csv") \
    .with_columns(
        pl.col("island").cast(pl.Categorical),
        pl.col("bill_length_mm").map_dict({"NA": None}, default = pl.col("bill_length_mm")),
        pl.col("flipper_length_mm").map_dict({"NA": None}, default = pl.col("flipper_length_mm")),) \
    .with_columns(
        pl.col("island").cast(pl.Int16),
        pl.col("bill_length_mm").cast(pl.Float32),
        pl.col("flipper_length_mm").cast(pl.Float32),
    )
mod = smf.mnlogit("island ~ bill_length_mm + flipper_length_mm", dat).fit()


def test_predictions_01():
    unknown = predictions(mod).filter(pl.col("group") == "0")
    known = pl.read_csv("tests/r/test_statsmodels_mnlogit_predictions_01.csv").filter(pl.col("group") == "Torgersen")
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-3)


def test_predictions_02():
    unknown = predictions(mod, by = "species").filter(pl.col("group") == "0")
    known = pl.read_csv("tests/r/test_statsmodels_mnlogit_predictions_02.csv").filter(pl.col("group") == "Torgersen")
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-4)


def test_comparisons_01():
    unknown = comparisons(mod).filter(pl.col("group") == "0")
    known = pl.read_csv("tests/r/test_statsmodels_mnlogit_comparisons_01.csv").filter(pl.col("group") == "Torgersen")
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-2)


def test_comparisons_02():
    unknown = comparisons(mod, by = "species").sort(["term", "species"]).filter(pl.col("group") == "0")
    known = pl.read_csv("tests/r/test_statsmodels_mnlogit_comparisons_02.csv").sort(["term", "species"]).filter(pl.col("group") == "Torgersen")
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-2)