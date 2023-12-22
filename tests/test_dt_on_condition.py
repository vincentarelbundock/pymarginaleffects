import polars as pl
import statsmodels.formula.api as smf

from marginaleffects import *
from marginaleffects.plot_common import dt_on_condition

from .utilities import *

df = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Guerry.csv", null_values = "NA") \
    .drop_nulls()
df = df.with_columns(pl.Series(range(df.shape[0])).alias("row_id")) \
    .sort("Region", "row_id")
mod = smf.ols("Literacy ~ Pop1831 * Desertion", df).fit()


def test_dt_on_condition():
    con = {'dept' : [1, 3] , 'Region' : "W", 'Department' : "Allier"}
    bp = dt_on_condition(mod, con)
    assert bp.shape[0] == 2
    assert (bp["Region"] == "W").all()
    con = "Area"
    assert dt_on_condition(mod, con).shape[0] == 100
    con = ["Area"]
    assert dt_on_condition(mod, con).shape[0] == 100
    con = ["Region", "Area"]
    assert dt_on_condition(mod, con).shape[0] == 25
    con = {"Region": None, "Department": "Allier"}
    assert dt_on_condition(mod, con).shape[0] == 5
    con = ["Region", "Area", "Pop1831"]
    assert dt_on_condition(mod, con).shape[0] == 125
    con = {"Area": None, "Region": "W"}
    assert dt_on_condition(mod, con).shape[0] == 100
