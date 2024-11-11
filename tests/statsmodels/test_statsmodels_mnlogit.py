import polars as pl
import pytest
import statsmodels.formula.api as smf
from polars.testing import assert_series_equal
from tests.conftest import penguins, penguins_with_nulls
from marginaleffects import *


dat = (
    penguins
    .drop_nulls(["species", "island", "bill_length_mm", "flipper_length_mm"])
    .with_columns(
        pl.col("island").cast(pl.Categorical).to_physical().cast(pl.Int8, strict=False),
        pl.col("flipper_length_mm").cast(pl.Float64),
    )
)
print(dat)
mod = smf.mnlogit("island ~ bill_length_mm + flipper_length_mm", dat.to_pandas()).fit()
r = {"0": "Torgersen", "1": "Biscoe", "2": "Dream"}


# @pytest.mark.skip(reason="statsmodels vcov is weird")
def test_predictions_01():
    """
    R code to generate the csv file data
    # Load necessary packages
    library(nnet)
    library(dplyr)
    library(marginaleffects)

    # Load and prepare the data
    penguins <- read.csv("https://vincentarelbundock.github.io/Rdatasets/csv/palmerpenguins/penguins.csv")
    penguins_clean <- penguins %>%
    select(island, bill_length_mm, flipper_length_mm) %>%
    na.omit()
    penguins_clean$island <- relevel(factor(penguins_clean$island), ref = "Biscoe")

    # Fit the model using nnet!
    model_r <- multinom(island ~ bill_length_mm  + flipper_length_mm, data = penguins_clean)

    # Extract and display coefficients in case the models do not match
    # coef_r <- coef(model_r)
    # print("Coefficients from R model:")
    # print(coef_r)

    predictions(model_r)

    """
    penguins_clean = penguins_with_nulls.select(['island', 'bill_length_mm', 'flipper_length_mm']).drop_nulls()

    # Define island categories and create a mapping
    island_categories = ["Biscoe", "Dream", "Torgersen"]
    island_mapping = {island: code for code, island in enumerate(island_categories)}

    # Map 'island' to integer codes
    penguins_clean = penguins_clean.with_columns(
        pl.col('island').replace_strict(island_mapping)
    )

    model_py = smf.mnlogit("island ~ bill_length_mm + flipper_length_mm", data=penguins_clean).fit()
    unknown = predictions(model_py).sort(by=["rowid", "group"])

    known = pl.read_csv("tests/r/test_statsmodels_mnlogit_predictions_03.csv")
    known = known.with_columns(pl.col("group").replace(island_mapping, return_dtype=pl.Int8).alias("group"))
    known = known.sort(by=["rowid", "group"])

    assert_series_equal(known["estimate"], unknown["estimate"], rtol=1e-2)


# @pytest.mark.skip(reason="statsmodels vcov is weird")
def test_predictions_02():
    unknown = predictions(mod, by="species")
    known = pl.read_csv("tests/r/test_statsmodels_mnlogit_predictions_02.csv")
    assert_series_equal(known["estimate"], unknown["estimate"], rtol=1e-2)


# @pytest.mark.skip(reason="statsmodels vcov is weird")
def test_comparisons_01():
    unknown = (
        comparisons(mod)
        .with_columns(pl.col("group").map_dict(r))
        .sort(["term", "group"])
    )
    known = pl.read_csv("tests/r/test_statsmodels_mnlogit_comparisons_01.csv").sort(
        ["term", "group"]
    )
    assert_series_equal(known["estimate"].head(), unknown["estimate"].head(), rtol=1e-1)

    unknown = comparisons(mod)
    known = pl.read_csv("tests/r/test_statsmodels_mnlogit_comparisons_01.csv")
    assert_series_equal(known["estimate"], unknown["estimate"], rtol=1e-2)


# @pytest.mark.skip(reason="statsmodels vcov is weird")
def test_comparisons_02():
    unknown = (
        comparisons(mod, by=["group", "species"])
        .with_columns(pl.col("group").map_dict(r))
        .sort(["term", "group", "species"])
    )
    known = pl.read_csv("tests/r/test_statsmodels_mnlogit_comparisons_02.csv").sort(
        ["term", "group", "species"]
    )
    assert_series_equal(known["estimate"], unknown["estimate"], rtol=1e-2)
