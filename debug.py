import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from pytest import approx

dat = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv") \
    .with_columns(pl.col("cyl").cast(pl.Utf8))
mod = smf.poisson("carb ~ mpg * qsec + cyl", data = dat).fit()
print(comparisons(mod, by = "cyl", vcov = False))