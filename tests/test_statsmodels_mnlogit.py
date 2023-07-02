import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from pytest import approx

dat = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/palmerpenguins/penguins.csv", null_values = "NA") \
    .drop_nulls(["species", "island", "bill_length_mm", "flipper_length_mm"]) \
    .with_columns(
        pl.col("island").cast(pl.Categorical).cast(pl.Int8),
        pl.col("flipper_length_mm").cast(pl.Float64)
    )

mod = smf.mnlogit("island ~ bill_length_mm + flipper_length_mm", dat.to_pandas()).fit()
r = {"0": "Torgersen", "1": "Biscoe", "2": "Dream"}


def test_predictions_01():
    unknown = predictions(mod).with_columns(pl.col("group").map_dict(r))
    known = pl.read_csv("tests/r/test_statsmodels_mnlogit_predictions_01.csv")
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-2)


def test_predictions_02():
    unknown = predictions(mod, by = "species")
    known = pl.read_csv("tests/r/test_statsmodels_mnlogit_predictions_02.csv")
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-2)


def test_comparisons_01():
    unknown = comparisons(mod)
    known = pl.read_csv("tests/r/test_statsmodels_mnlogit_comparisons_01.csv")
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-2)


def test_comparisons_02():
    unknown = comparisons(mod, by = ["group", "species"]) \
        .with_columns(pl.col("group").map_dict(r)) \
        .sort(["term", "group", "species"])
    known = pl.read_csv("tests/r/test_statsmodels_mnlogit_comparisons_02.csv") \
        .sort(["term", "group", "species"])
    assert known["estimate"].to_numpy() == approx(unknown["estimate"].to_numpy(), rel = 1e-1)

# (known["estimate"] - unknown["estimate"]).abs() / known["estimate"]
# known.select(["group", "term", "species", "estimate"])
# unknown.select(["group", "term", "species", "estimate"])