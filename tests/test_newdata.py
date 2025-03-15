import polars as pl
import statsmodels.formula.api as smf
from tests.helpers import mtcars
from marginaleffects import *

mod = smf.probit("am ~ hp + wt", data=mtcars).fit()


def test_mean_median():
    a = slopes(mod, newdata="mean")
    b = slopes(mod, newdata="median")
    assert a.shape[0] == 2
    assert b.shape[0] == 2
    assert all(b["estimate"].to_numpy() != a["estimate"].to_numpy())


def test_predictions_mean():
    p = predictions(mod, newdata="median")
    assert p.shape[0] == 1


def test_predictions_padding():
    dat = pl.read_csv("tests/data/impartiality.csv").with_columns(
        pl.col("impartial").cast(pl.Int32)
    )
    m = smf.logit(
        "impartial ~ equal * democracy + continent", data=dat.to_pandas()
    ).fit()
    p = predictions(m, newdata=dat.head())
    assert p.shape[0] == 5
    p = predictions(m, newdata="mean")
    assert p.shape[0] == 1
