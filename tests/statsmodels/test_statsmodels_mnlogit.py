import polars as pl
import pytest
import statsmodels.formula.api as smf
from polars.testing import assert_series_equal
from tests.conftest import penguins, penguins_with_nulls
from marginaleffects import *


island_categories = ["Biscoe", "Dream", "Torgersen"]
island_mapping = {island: code for code, island in enumerate(island_categories)}
dat = penguins.drop_nulls(
    ["species", "island", "bill_length_mm", "flipper_length_mm"]
).with_columns(
    pl.col("island").replace_strict(island_mapping),
    pl.col("flipper_length_mm").cast(pl.Float64),
)
mod = smf.mnlogit("island ~ bill_length_mm + flipper_length_mm", dat.to_pandas()).fit()


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
    penguins_clean = penguins_with_nulls.select(
        ["island", "bill_length_mm", "flipper_length_mm"]
    ).drop_nulls()

    # Define island categories and create a mapping
    island_categories = ["Biscoe", "Dream", "Torgersen"]
    island_mapping = {island: code for code, island in enumerate(island_categories)}

    # Map 'island' to integer codes
    penguins_clean = penguins_clean.with_columns(
        pl.col("island").replace_strict(island_mapping)
    )

    model_py = smf.mnlogit(
        "island ~ bill_length_mm + flipper_length_mm", data=penguins_clean
    ).fit()
    unknown = predictions(model_py).sort(by=["rowid", "group"])

    known = pl.read_csv("tests/r/test_statsmodels_mnlogit_predictions_01.csv")
    known = known.with_columns(
        pl.col("group").replace(island_mapping, return_dtype=pl.Int8).alias("group")
    )
    known = known.sort(by=["rowid", "group"])

    assert_series_equal(known["estimate"], unknown["estimate"], rtol=1e-2)


# @pytest.mark.skip(reason="statsmodels vcov is weird")
def test_predictions_02():
    unknown = predictions(mod, by="species").sort(["group", "species"])
    known = (
        pl.read_csv("tests/r/test_statsmodels_mnlogit_predictions_02.csv")
        .with_columns(pl.col("group").replace_strict(island_mapping))
        .sort(["group", "species"])
    )
    assert_series_equal(known["estimate"], unknown["estimate"], rtol=1e-1)


def test_comparisons_01():
    penguins_clean = penguins_with_nulls.select(
        ["island", "bill_length_mm", "flipper_length_mm"]
    ).drop_nulls()

    # Define island categories and create a mapping
    island_categories = ["Biscoe", "Dream", "Torgersen"]
    island_mapping = {island: code for code, island in enumerate(island_categories)}

    # Map 'island' to integer codes
    penguins_clean = penguins_clean.with_columns(
        pl.col("island").replace_strict(island_mapping)
    )

    mod = smf.mnlogit(
        "island ~ bill_length_mm + flipper_length_mm", data=penguins_clean
    ).fit()
    unknown = (
        comparisons(mod)
        .with_columns(pl.col("group").replace(island_mapping))
        .sort(["rowid", "term", "group"])
    )
    known = (
        pl.read_csv("tests/r/test_statsmodels_mnlogit_comparisons_01.csv")
        .with_columns(pl.col("group").replace(island_mapping))
        .sort(["rowid", "term", "group"])
    )
    new_column_names = {col: col.replace('.', '_') for col in known.columns}
    known = known.rename(new_column_names)
    print(known.head())
    print(unknown.head())
    print(compare_polars_tables(known, unknown, index=0))
    assert_series_equal(known["estimate"].head(), unknown["estimate"].head(), rtol=2)

    unknown = comparisons(mod)
    known = pl.read_csv("tests/r/test_statsmodels_mnlogit_comparisons_01.csv")
    assert_series_equal(known["estimate"], unknown["estimate"], rtol=1e-2)

# Function to print visual comparison
def compare_polars_tables(known, unknown, index=0):
    headers = ["Column", "Table known Value", "Table unknown Value", "Difference"]
    row_format = "{:<25} {:<25} {:<25} {:<15}"
    
    print(row_format.format(*headers))
    print("-" * 60)
    
    for col in known.columns:
        val1 = known[col][index]
        val2 = unknown[col][index]
        difference = "Yes" if val1 != val2 else "No"
        
        print(row_format.format(col, val1, val2, difference))
    print(row_format.format("index", index, index, "No"))

# # @pytest.mark.skip(reason="statsmodels vcov is weird")
# def test_comparisons_02():
#     unknown = (
#         comparisons(mod, by=["group", "species"])
#         .with_columns(pl.col("group").map_dict(r))
#         .sort(["term", "group", "species"])
#     )
#     known = pl.read_csv("tests/r/test_statsmodels_mnlogit_comparisons_02.csv").sort(
#         ["term", "group", "species"]
#     )
#     assert_series_equal(known["estimate"], unknown["estimate"], rtol=1e-2)
