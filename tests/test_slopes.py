import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from scipy.stats import pearsonr
from pytest import approx
from marginaleffects import *
from marginaleffects.testing import *
from rpy2.robjects.packages import importr
from marginaleffects.comparisons import estimands
import matplotlib.pyplot as plt

# R packages
marginaleffects = importr("marginaleffects")
stats = importr("stats")

# Guerry Data
df, df_r = rdatasets("HistData", "Guerry", r = True)
mod_py = smf.ols("Literacy ~ Pop1831 * Desertion", df).fit()
mod_r = stats.lm("Literacy ~ Pop1831 * Desertion", data = df_r)


# mtcars Data
df_py, df_r = rdatasets("datasets", "mtcars", r = True)
mod_py = smf.ols("mpg ~ wt * hp", df_py).fit()
mod_r = stats.lm("mpg ~ wt * hp", data = df_r)


# Slopes
cmp_py = slopes(mod_py, slope = "dydx", by = False)
cmp_r = marginaleffects.slopes(mod_r, slope = "dydx", eps = 1e-4)
cmp_r = r_to_polars(cmp_r)


def test_comparison_derivatives():
    est = [k for k in estimands.keys() if re.search("x", k) is not None]
    a = ["dydxavg", "eydxavg", "eyexavg", "dyexavg"]
    b = ["dydx", "eydx", "dyex", "eyex"]
    est = a + b
    for e in est:
        cmp_py = comparisons(mod_py, comparison = e)
        cmp_r = marginaleffects.slopes(mod_r, slope = e, eps = 1e-4)
        cmp_r = r_to_polars(cmp_r)
        compare_r_to_py(cmp_r, cmp_py, tolr = 1e-1, tola = 2e-2, msg = e)