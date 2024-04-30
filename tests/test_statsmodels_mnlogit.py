# import polars as pl
# import pytest
# import statsmodels.formula.api as smf
# from polars.testing import assert_series_equal

# from marginaleffects import *

# @pytest.mark.skip(reason="TODO: check this")
# dat = (
#     pl.read_csv(
#         "https://vincentarelbundock.github.io/Rdatasets/csv/palmerpenguins/penguins.csv",
#         null_values="NA",
#     )
#     .drop_nulls(["species", "island", "bill_length_mm", "flipper_length_mm"])
#     .with_columns(
#         pl.col("island").cast(pl.Categorical, strict=False).cast(pl.Int8, strict=False),
#         pl.col("flipper_length_mm").cast(pl.Float64),
#     )
# )

# mod = smf.mnlogit("island ~ bill_length_mm + flipper_length_mm", dat.to_pandas()).fit()
# r = {"0": "Torgersen", "1": "Biscoe", "2": "Dream"}


# @pytest.mark.skip(reason="statsmodels vcov is weird")
# def test_predictions_01():
#     unknown = predictions(mod).with_columns(pl.col("group").map_dict(r))
#     known = pl.read_csv("tests/r/test_statsmodels_mnlogit_predictions_01.csv")
#     assert_series_equal(known["estimate"], unknown["estimate"], rtol=1e-2)


# @pytest.mark.skip(reason="statsmodels vcov is weird")
# def test_predictions_02():
#     unknown = predictions(mod, by="species")
#     known = pl.read_csv("tests/r/test_statsmodels_mnlogit_predictions_02.csv")
#     assert_series_equal(known["estimate"], unknown["estimate"], rtol=1e-2)


# @pytest.mark.skip(reason="statsmodels vcov is weird")
# def test_comparisons_01():
#     unknown = (
#         comparisons(mod)
#         .with_columns(pl.col("group").map_dict(r))
#         .sort(["term", "group"])
#     )
#     known = pl.read_csv("tests/r/test_statsmodels_mnlogit_comparisons_01.csv").sort(
#         ["term", "group"]
#     )
#     assert_series_equal(known["estimate"].head(), unknown["estimate"].head(), rtol=1e-1)

#     unknown = comparisons(mod)
#     known = pl.read_csv("tests/r/test_statsmodels_mnlogit_comparisons_01.csv")
#     assert_series_equal(known["estimate"], unknown["estimate"], rtol=1e-2)


# @pytest.mark.skip(reason="statsmodels vcov is weird")
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
