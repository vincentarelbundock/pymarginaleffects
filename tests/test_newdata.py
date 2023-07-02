from marginaleffects import *
import statsmodels.formula.api as smf
import polars as pl

mtcars = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv")

def test_mean_median():
    mod = smf.probit("am ~ hp + wt", data = mtcars).fit()
    a = slopes(mod, newdata = "mean")
    b = slopes(mod, newdata = "median")
    assert a.shape[0] == 2
    assert b.shape[0] == 2
    assert all(b["estimate"].to_numpy() != a["estimate"].to_numpy())
