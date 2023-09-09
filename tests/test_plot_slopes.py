import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from marginaleffects.plot_slopes import *
from .utilities import *

df = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/palmerpenguins/penguins.csv", null_values = "NA") \
    .drop_nulls()
mod = smf.ols("body_mass_g ~ flipper_length_mm * species * bill_length_mm + island", df).fit()

mtcars = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv")
mod_cars = smf.ols("mpg ~ wt * qsec * gear", mtcars).fit()

def test_plot_slopes():

    # book

    bp = plot_slopes(mod,
      variables = "bill_length_mm",
      slope = "eyex",
      condition = ['species', 'island'])
    assert hasattr(bp, "show")

    bp = plot_slopes(mod,
      variables = "bill_length_mm",
      slope = "eyex",
      by = ['species', 'island'])
    assert hasattr(bp, "show")

    bp = plot_slopes(mod_cars, variables = "qsec", condition = ["wt", "gear"])
    assert hasattr(bp, "show")