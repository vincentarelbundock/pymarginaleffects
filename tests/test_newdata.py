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


def test_predictions_mean():
    p = predictions(mod, newdata = "median")
    assert p.shape[0] == 1


dat = pl.read_csv("tests/data/impartiality.csv").with_columns(pl.col("impartial").cast(pl.Int32))
mod = smf.logit("impartial ~ equal * democracy + continent", data = dat).fit()

p = predictions(mod, newdata = dat.head())
print(p)

p = predictions(mod, newdata = "mean")
print(p)