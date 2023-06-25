import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from pytest import approx
from marginaleffects import *
from marginaleffects.testing import *
from rpy2.robjects.packages import importr

# R packages
marginaleffects = importr("marginaleffects")
stats = importr("stats")

# Guerry Data
df, df_r = rdatasets("HistData", "Guerry", r = True)
df = df.with_columns((pl.col("Area") > pl.col("Area").median()).alias("Bool"))
df = df.with_columns((pl.col("Distance") > pl.col("Distance").median()).alias("Bin"))
df = df.with_columns(df['Bin'].apply(lambda x: int(x), return_dtype=pl.Int32).alias('Bin'))
df = df.with_columns(pl.Series(np.random.choice(["a", "b", "c"], df.shape[0])).alias("Char"))

def test_basic():
    mod_py = smf.ols("Literacy ~ Pop1831 * Desertion", df).fit()
    mod_r = stats.lm("Literacy ~ Pop1831 * Desertion", data = df_r)
    cmp_py = comparisons(mod_py, comparison = "differenceavg")
    cmp_r = marginaleffects.comparisons(mod_r, comparison = "differenceavg")
    cmp_r = r_to_polars(cmp_r)
    compare_r_to_py(cmp_r, cmp_py)


def test_HC3():
    mod_py = smf.ols("Literacy ~ Pop1831 * Desertion", df).fit()
    mod_r = stats.lm("Literacy ~ Pop1831 * Desertion", data = df_r)
    cmp_py = comparisons(mod_py, comparison = "differenceavg", vcov = "HC3")
    cmp_r = marginaleffects.comparisons(mod_r, comparison = "differenceavg", vcov = "HC3")
    cmp_r = r_to_polars(cmp_r)
    compare_r_to_py(cmp_r, cmp_py)


def test_bare_minimum():
    fit = smf.ols("Literacy ~ Pop1831 * Desertion + Bool + Bin + Char", df).fit()
    assert type(comparisons(fit)) == pl.DataFrame
    assert type(comparisons(fit, variables = "Pop1831", comparison = "differenceavg")) == pl.DataFrame
    assert type(comparisons(fit, variables = "Pop1831", comparison = "difference").head()) == pl.DataFrame
    assert type(comparisons(fit, variables = "Pop1831", comparison = "ratio").head()) == pl.DataFrame
    assert type(comparisons(fit, variables = "Pop1831", comparison = "difference", by = "Region")) == pl.DataFrame
    assert type(comparisons(fit, vcov = False, comparison = "differenceavg")) == pl.DataFrame
    assert type(comparisons(fit, vcov = "HC3", comparison = "differenceavg")) == pl.DataFrame
    assert type(comparisons(fit)) == pl.DataFrame
    assert type(comparisons(fit, variables = {"Char": "sequential"})) == pl.DataFrame
    assert type(comparisons(fit, variables = "Pop1831")) == pl.DataFrame
    assert type(comparisons(fit, variables = ["Pop1831", "Desertion"])) == pl.DataFrame
    assert type(comparisons(fit, variables = {"Pop1831": 1000, "Desertion": 2})) == pl.DataFrame
    assert type(comparisons(fit, variables = {"Pop1831": [100, 2000]})) == pl.DataFrame