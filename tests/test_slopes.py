import statsmodels.formula.api as smf
import numpy as np
from scipy.stats import pearsonr
from pytest import approx
from marginaleffects import *
from .utilities import *
from rpy2.robjects.packages import importr
from marginaleffects.comparisons import estimands
import matplotlib.pyplot as plt
from polars.testing import assert_series_equal

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


def test_comparison_derivatives():
    est = [k for k in estimands.keys() if re.search("x", k) is not None]
    a = ["dydxavg", "eydxavg", "eyexavg", "dyexavg"]
    b = ["dydx", "eydx", "eyex", "dyex"]
    est = a + b
    for e in est:
        cmp_py = comparisons(mod_py, comparison = e)
        cmp_r = marginaleffects.slopes(mod_r, slope = e, eps = 1e-4)
        cmp_r = r_to_polars(cmp_r)
        cols = [x for x in ["term", "contrast"] if x in cmp_r.columns]
        cmp_py = cmp_py.sort(cols)
        cmp_r = cmp_r.sort(cols)
        assert_series_equal(cmp_r["estimate"], cmp_py["estimate"], check_names = False, rtol = 1e-2)
        # assert_series_equal(cmp_r["std.error"], cmp_py["std_error"], check_names = False, rtol = 1e-2)