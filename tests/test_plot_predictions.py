import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from marginaleffects.plot import * 
from .utilities import *


df = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Guerry.csv", null_values = "NA") \
    .drop_nulls()
df = df.with_columns(pl.Series(range(df.shape[0])).alias("row_id")) \
    .sort("Region", "row_id")
mod = smf.ols("Literacy ~ Pop1831 * Desertion + Region + Area", df).fit()


def test_plot_predictions():

    con = {'Region' : None, 'Area' : None, 'Desertion' : [0, 30, 90]}
    bp = plot_predictions(mod, con)

    con = {'Pop1831' : None, 'Area' : None, 'Desertion' : [0, 30, 90]}
    bp = plot_predictions(mod, con)

    con = {'Region' : None, 'Desertion' : [0, 30, 90]}
    bp = plot_predictions(mod, con)

    con = {'Pop1831' : None, 'Desertion' : [0, 30, 90]}
    bp = plot_predictions(mod, con)

    con = {'Region' : None}
    bp = plot_predictions(mod, con)

    con = {'Pop1831' : None}
    bp = plot_predictions(mod, con)

    con = ['Region', 'Area', 'Desertion']
    bp = plot_predictions(mod, con)

    con = ['Pop1831', 'Area', 'Desertion']
    bp = plot_predictions(mod, con)

    con = ['Region', 'Area']
    bp = plot_predictions(mod, con)

    con = ['Pop1831', 'Area']
    bp = plot_predictions(mod, con)

    con = ['Region']
    bp = plot_predictions(mod, con)

    con = ['Pop1831']
    bp = plot_predictions(mod, con)

    con = 'Region'
    bp = plot_predictions(mod, con)

    con = 'Pop1831'
    bp = plot_predictions(mod, con)