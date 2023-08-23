import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from marginaleffects.plot import *
from .utilities import *


df = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Guerry.csv", null_values = "NA") \
    .drop_nulls()
df = df.with_columns(pl.Series(range(df.shape[0])).alias("row_id")) \
    .sort("Region", "row_id")
con_mod = smf.ols("Literacy ~ Pop1831 * Desertion + Region + Area + MainCity", df).fit()

df = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/palmerpenguins/penguins.csv", null_values = "NA") \
    .drop_nulls()
by_mod = smf.ols("body_mass_g ~ flipper_length_mm * species * bill_length_mm + island", df).fit()

def test_plot_predictions():

    # by

    bp = plot_predictions(by_mod, by='bill_length_mm')
    assert hasattr(bp, "show")

    bp = plot_predictions(by_mod, by='bill_length_mm', newdata=datagrid(by_mod, bill_length_mm=[37,39]))
    assert hasattr(bp, "show")

    bp = plot_predictions(by_mod, by=['bill_length_mm'])
    assert hasattr(bp, "show")

    bp = plot_predictions(by_mod, by=['bill_length_mm'], newdata=datagrid(by_mod, bill_length_mm=[37,39]))
    assert hasattr(bp, "show")

    bp = plot_predictions(by_mod, by=['bill_length_mm', 'island'])
    assert hasattr(bp, "show")

    bp = plot_predictions(by_mod, by=['bill_length_mm', 'island'], newdata=datagrid(by_mod, bill_length_mm=[37,39]))
    assert hasattr(bp, "show")

    bp = plot_predictions(by_mod, by=['bill_length_mm', 'island', 'species'])
    assert hasattr(bp, "show")

    bp = plot_predictions(by_mod, by=['bill_length_mm', 'island', 'species'], newdata=datagrid(by_mod, bill_length_mm=[37,39]))
    assert hasattr(bp, "show")



    # condition

    con = {'Region' : None, 'Area' : None, 'Desertion' : [0, 30, 90]}
    bp = plot_predictions(con_mod, condition=con)
    assert hasattr(bp, "show")

    con = {'Pop1831' : None, 'Area' : None, 'Desertion' : [0, 30, 90]}
    bp = plot_predictions(con_mod, condition=con)
    assert hasattr(bp, "show")

    con = {'Region' : None, 'Desertion' : [0, 30, 90]}
    bp = plot_predictions(con_mod, condition=con)
    assert hasattr(bp, "show")

    con = {'Pop1831' : None, 'Desertion' : [0, 30, 90]}
    bp = plot_predictions(con_mod, condition=con)
    assert hasattr(bp, "show")

    con = {'Region' : None}
    bp = plot_predictions(con_mod, condition=con)
    assert hasattr(bp, "show")

    con = {'Pop1831' : None}
    bp = plot_predictions(con_mod, condition=con)
    assert hasattr(bp, "show")

    con = ['Region', 'Area', 'Desertion']
    bp = plot_predictions(con_mod, condition=con)
    assert hasattr(bp, "show")

    con = ['Pop1831', 'Area', 'Desertion']
    bp = plot_predictions(con_mod, condition=con)
    assert hasattr(bp, "show")

    con = ['Region', 'Area']
    bp = plot_predictions(con_mod, condition=con)
    assert hasattr(bp, "show")

    con = ['Pop1831', 'Area']
    bp = plot_predictions(con_mod, condition=con)
    assert hasattr(bp, "show")

    con = ['Region']
    bp = plot_predictions(con_mod, condition=con)
    assert hasattr(bp, "show")

    con = ['Pop1831']
    bp = plot_predictions(con_mod, condition=con)
    assert hasattr(bp, "show")

    con = 'Region'
    bp = plot_predictions(con_mod, condition=con)
    assert hasattr(bp, "show")

    con = 'Pop1831'
    bp = plot_predictions(con_mod, condition=con)
    assert hasattr(bp, "show")