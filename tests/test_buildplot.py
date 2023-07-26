import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from marginaleffects.buildplot import build_plot
from .utilities import *


df = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Guerry.csv", null_values = "NA") \
    .drop_nulls() \
    .with_columns(pl.Series(range(df.shape[0])).alias("row_id")) \
    .sort("Region", "row_id")
mod = smf.ols("Literacy ~ Pop1831 * Desertion", df).fit()


def test_build_plot():
    con = {'dept' : [1, 3] , 'Region' : "W", 'Department' : "Allier"}
    assert build_plot(mod, con).shape[0] == 2
    con = "Area"
    assert build_plot(mod, con).shape[0] == 100
    con = ["Area"]
    assert build_plot(mod, con).shape[0] == 100
    con = {"Area": None, "Region": "W"}
    assert build_plot(mod, con).shape[0] == 100
    con = ["Region", "Area"]
    assert build_plot(mod, con).shape[0] == 25
    con = ["Region", "Area", "Pop1831"]
    assert build_plot(mod, con).shape[0] == 125





           
           ["Distance", "Region"])