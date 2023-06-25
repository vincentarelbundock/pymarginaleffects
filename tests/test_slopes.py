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
df_py, df_r = download_data("datasets", "mtcars")
mod_py = smf.ols("mpg ~ wt * hp", df_py).fit()
mod_r = stats.lm("mpg ~ wt * hp", data = df_r)

def test_dydx():
    slo_r = marginaleffects.slopes(mod_r, slope = "dydxavg")
    slo_r = r_to_polars(slo_r)
    slo_py = comparisons(mod_py, comparison = "dydxavg", newdata = df_py)
    compare_r_to_py(slo_r, slo_py)
    slo_r = marginaleffects.slopes(mod_r, slope = "dydx")
    slo_r = r_to_polars(slo_r)
    slo_py = comparisons(mod_py, comparison = "dydx", newdata = df_py)
    slo_r["estimate"] - slo_py["estimate"]
    compare_r_to_py(slo_r, slo_py, rel = 1e-1)
