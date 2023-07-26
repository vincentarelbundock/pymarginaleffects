import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from marginaleffects.buildplot import build_plot
from .utilities import *


df = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Guerry.csv", null_values = "NA").drop_nulls()
df = df \
    .with_columns(pl.Series(range(df.shape[0])).alias("row_id")) \
    .sort("Region", "row_id")
mod_py = smf.ols("Literacy ~ Pop1831 * Desertion", df).fit()

def test_build_plot():
    con_list = ['dept', 'Region']
    con_dict = {'dept' : [1,3] , 'Region' : 1, 'Department' : 2}
    build_plot(mod_py, con_list)
