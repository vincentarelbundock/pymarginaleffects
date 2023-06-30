import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from pytest import approx
from marginaleffects import *
from .utilities import *
from rpy2.robjects.packages import importr
from marginaleffects.comparisons import estimands

# R packages
marginaleffects = importr("marginaleffects")
stats = importr("stats")

# Guerry Data
df, df_r = rdatasets("HistData", "Guerry", r = True)
df = df \
    .with_columns(
        (pl.col("Area") > pl.col("Area").median()).alias("Bool"),
        (pl.col("Distance") > pl.col("Distance").median()).alias("Bin")) \
    .with_columns(
        pl.col("Bin").apply(lambda x: int(x), return_dtype=pl.Int32).alias('Bin'),
        pl.Series(np.random.choice(["a", "b", "c"], df.shape[0])).alias("Char"))
mod_py = smf.ols("Literacy ~ Pop1831 * Desertion", df).fit()
mod_r = stats.lm("Literacy ~ Pop1831 * Desertion", data = df_r)


def test_predictions():
    cmp_py = predictions(mod_py)
    cmp_r = marginaleffects.predictions(mod_r)
    cmp_r = r_to_polars(cmp_r)
    compare_r_to_py(cmp_r, cmp_py)


def test_by():
    cmp_py = predictions(mod_py, by = "Region")
    cmp_r = marginaleffects.predictions(mod_r, by = "Region")
    cmp_r = r_to_polars(cmp_r)
    compare_r_to_py(cmp_r, cmp_py)


def test_by_hypothesis():
    cmp_py = predictions(mod_py, by = "Region", hypothesis = "b1 * b3 = b2**2")
    cmp_r = marginaleffects.predictions(mod_r, by = "Region", hypothesis = "b1 * b3 = b2^2")
    cmp_r = r_to_polars(cmp_r)
    compare_r_to_py(cmp_r, cmp_py)