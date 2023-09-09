import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from marginaleffects.plot_comparisons import *
from .utilities import *


df = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/palmerpenguins/penguins.csv", null_values = "NA") \
    .drop_nulls()
mod = smf.ols("body_mass_g ~ flipper_length_mm * species * bill_length_mm + island", df).fit()

def test_plot_comparisons():

    # book
    bp = plot_comparisons(mod, variables = "flipper_length_mm", condition = ["bill_length_mm", "species"])
    assert hasattr(bp, "show")

    bp = plot_comparisons(mod, variables = {'flipper_length_mm' : 'sd'}, condition = ["bill_length_mm", "species"])
    assert hasattr(bp, "show")

    bp = plot_comparisons(mod, variables = {"flipper_length_mm" : 10}, condition = ["bill_length_mm", "species"])
    assert hasattr(bp, "show")

    bp = plot_comparisons(mod, variables = "species", condition = "bill_length_mm", comparison = "ratio")
    assert hasattr(bp, "show")

    bp = plot_comparisons(mod, variables = "species", condition = "bill_length_mm")
    assert hasattr(bp, "show")

    bp = plot_comparisons(mod, variables = "flipper_length_mm", by = "species")
    assert hasattr(bp, "show")

    bp = plot_comparisons(mod, variables = "flipper_length_mm", by = ["species", "island"])
    assert hasattr(bp, "show")

    bp = plot_comparisons(mod, variables = ["flipper_length_mm", "bill_length_mm"], by = ["species", "island"])
    assert hasattr(bp, "show")