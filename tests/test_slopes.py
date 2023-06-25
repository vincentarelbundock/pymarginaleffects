#v TODO: bad tolerance

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
df_py, df_r = rdatasets("datasets", "mtcars", r = True)
mod_py = smf.ols("mpg ~ wt * hp", df_py).fit()
mod_r = stats.lm("mpg ~ wt * hp", data = df_r)


def test_dydx():
    slo_r = marginaleffects.slopes(mod_r, slope = "dydxavg", eps = 1e-4)
    slo_r = r_to_polars(slo_r)
    slo_py = comparisons(mod_py, comparison = "dydxavg", newdata = df_py, eps = 1e-4)
    compare_r_to_py(slo_r, slo_py)
    slo_r = marginaleffects.slopes(mod_r, slope = "dydx", eps = 1e-4)
    slo_r = r_to_polars(slo_r)
    slo_py = comparisons(mod_py, comparison = "dydx", newdata = df_py, eps = 1e-4)
    compare_r_to_py(slo_r, slo_py, rel = 2e-2)


# def test_eyex():
#     slo_r = marginaleffects.slopes(mod_r, slope = "eyexavg", eps = 1e-4)
#     slo_r = r_to_polars(slo_r)
#     slo_py = comparisons(mod_py, comparison = "eyexavg", newdata = df_py, eps = 1e-4)
#     compare_r_to_py(slo_r, slo_py, rel = 2e-2)
#     slo_r = marginaleffects.slopes(mod_r, slope = "eyex", eps = 1e-4)
#     slo_r = r_to_polars(slo_r)
#     slo_py = comparisons(mod_py, comparison = "eyex", newdata = df_py, eps = 1e-4)
#     compare_r_to_py(slo_r, slo_py, rel = 2e-2)


# def test_eydx():
#     slo_r = marginaleffects.slopes(mod_r, slope = "eydxavg", eps = 1e-4)
#     slo_r = r_to_polars(slo_r)
#     slo_py = comparisons(mod_py, comparison = "eydxavg", newdata = df_py, eps = 1e-4)
#     compare_r_to_py(slo_r, slo_py, rel = 2e-2)
#     slo_r = marginaleffects.slopes(mod_r, slope = "eydx", eps = 1e-4)
#     slo_r = r_to_polars(slo_r)
#     slo_py = comparisons(mod_py, comparison = "eydx", newdata = df_py, eps = 1e-4)
#     compare_r_to_py(slo_r, slo_py, rel = 2e-2)


def test_dyex():
    slo_r = marginaleffects.slopes(mod_r, slope = "dyexavg", eps = 1e-4)
    slo_r = r_to_polars(slo_r)
    slo_py = comparisons(mod_py, comparison = "dyexavg", newdata = df_py, eps = 1e-4)
    compare_r_to_py(slo_r, slo_py, rel = 2e-2)
    slo_r = marginaleffects.slopes(mod_r, slope = "dyex", eps = 1e-4)
    slo_r = r_to_polars(slo_r)
    slo_py = comparisons(mod_py, comparison = "dyex", newdata = df_py, eps = 1e-4)
    compare_r_to_py(slo_r, slo_py, rel = 2e-2)