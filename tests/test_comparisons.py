import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
import polars as pl
import numpy as np
from pytest import approx
from marginaleffects import *
import marginaleffects
from marginaleffects.comparisons import estimands
from polars.testing import assert_series_equal


dat = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Guerry.csv", null_values = "NA") \
    .drop_nulls() \
    .with_columns(
        (pl.col("Area") > pl.col("Area").median()).alias("Bool"),
        (pl.col("Distance") > pl.col("Distance").median()).alias("Bin")) 
dat = dat \
    .with_columns(
        pl.col("Bin").apply(lambda x: int(x), return_dtype=pl.Int32).alias('Bin'),
        pl.Series(np.random.choice(["a", "b", "c"], dat.shape[0])).alias("Char"))
mod = smf.ols("Literacy ~ Pop1831 * Desertion", dat).fit()


def test_difference():
    cmp_py = comparisons(mod, comparison = "differenceavg").sort("term")
    cmp_r = pl.read_csv("tests/r/test_comparisons_01.csv").sort("term")
    assert_series_equal(cmp_py["estimate"], cmp_r["estimate"])
    assert_series_equal(cmp_py["std_error"], cmp_r["std.error"], check_names = False, rtol = 1e-3)
    cmp_py = comparisons(mod, comparison = "difference").sort("term", "rowid")
    cmp_r = pl.read_csv("tests/r/test_comparisons_02.csv").sort("term", "rowid")
    assert_series_equal(cmp_py["estimate"], cmp_r["estimate"])
    assert_series_equal(cmp_py["std_error"], cmp_r["std.error"], check_names = False, rtol = 1e-3)


def test_comparison_simple():
    est = [k for k in estimands.keys() if not re.search("x|wts", k)]
    for e in est:
        cmp_py = comparisons(mod, comparison = e).sort("term")
        cmp_r = pl.read_csv(f"tests/r/test_comparisons_03_{e}.csv").sort("term")
        if cmp_r.shape[1] == 170:
            raise ValueError("R and Python results are not the same")
        assert_series_equal(cmp_py["estimate"], cmp_r["estimate"], rtol = 1e-2)
        assert_series_equal(cmp_py["std_error"], cmp_r["std.error"], check_names = False, rtol = 3e-2)


def test_by():
    cmp_py = comparisons(mod, comparison = "differenceavg", by = "Region").sort("term", "Region")
    cmp_r = pl.read_csv("tests/r/test_comparisons_04.csv").sort("term", "Region")
    assert_series_equal(cmp_py["estimate"], cmp_r["estimate"])
    assert_series_equal(cmp_py["std_error"], cmp_r["std.error"], check_names = False, rtol = 1e-3)


def test_HC3():
    cmp_py = comparisons(mod, comparison = "differenceavg", vcov = "HC3").sort("term")
    cmp_r = pl.read_csv("tests/r/test_comparisons_05.csv").sort("term")
    assert_series_equal(cmp_py["estimate"], cmp_r["estimate"])
    assert_series_equal(cmp_py["std_error"], cmp_r["std.error"], check_names = False, rtol = 1e-3)


def test_difference_wts():
    cmp_py = comparisons(mod, variables = "Desertion", by = "Region", wts = "Literacy")
    cmp_r = pl.read_csv("tests/r/test_comparisons_06.csv").sort("Region")
    assert_series_equal(cmp_py["estimate"], cmp_r["estimate"])
    assert_series_equal(cmp_py["std_error"], cmp_r["std.error"], check_names = False, rtol = 1e-4)
    cmp_py = comparisons(mod, variables = "Desertion", by = "Region")
    cmp_r = pl.read_csv("tests/r/test_comparisons_07.csv").sort("Region")
    assert_series_equal(cmp_py["estimate"], cmp_r["estimate"])
    assert_series_equal(cmp_py["std_error"], cmp_r["std.error"], check_names = False, rtol = 1e-4)


def test_bare_minimum():
    fit = smf.ols("Literacy ~ Pop1831 * Desertion + Bool + Bin + Char", dat).fit()
    assert type(comparisons(fit)) == marginaleffects.classes.MarginaleffectsDataFrame
    assert type(comparisons(fit, variables = "Pop1831", comparison = "differenceavg")) == marginaleffects.classes.MarginaleffectsDataFrame
    assert type(comparisons(fit, variables = "Pop1831", comparison = "difference").head()) == marginaleffects.classes.MarginaleffectsDataFrame
    assert type(comparisons(fit, variables = "Pop1831", comparison = "ratio").head()) == marginaleffects.classes.MarginaleffectsDataFrame
    assert type(comparisons(fit, variables = "Pop1831", comparison = "difference", by = "Region")) == marginaleffects.classes.MarginaleffectsDataFrame
    assert type(comparisons(fit, vcov = False, comparison = "differenceavg")) == marginaleffects.classes.MarginaleffectsDataFrame
    assert type(comparisons(fit, vcov = "HC3", comparison = "differenceavg")) == marginaleffects.classes.MarginaleffectsDataFrame
    assert type(comparisons(fit)) == marginaleffects.classes.MarginaleffectsDataFrame
    assert type(comparisons(fit, variables = {"Char": "sequential"})) == marginaleffects.classes.MarginaleffectsDataFrame
    assert type(comparisons(fit, variables = "Pop1831")) == marginaleffects.classes.MarginaleffectsDataFrame
    assert type(comparisons(fit, variables = ["Pop1831", "Desertion"])) == marginaleffects.classes.MarginaleffectsDataFrame
    assert type(comparisons(fit, variables = {"Pop1831": 1000, "Desertion": 2})) == marginaleffects.classes.MarginaleffectsDataFrame
    assert type(comparisons(fit, variables = {"Pop1831": [100, 2000]})) == marginaleffects.classes.MarginaleffectsDataFrame