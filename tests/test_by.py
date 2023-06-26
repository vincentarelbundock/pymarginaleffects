import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from pytest import approx
from marginaleffects import *
from marginaleffects.testing import *
from rpy2.robjects.packages import importr
from marginaleffects.comparisons import estimands

# R packages
marginaleffects = importr("marginaleffects")
stats = importr("stats")
df, df_r = rdatasets("HistData", "Guerry", r = True)
mod_py = smf.ols("Literacy ~ Pop1831 * Desertion", df).fit()
mod_r = stats.lm("Literacy ~ Pop1831 * Desertion", data = df_r)


def test_predictions_by_string():
    cmp_py = predictions(mod_py, by = "Region")
    cmp_r = marginaleffects.predictions(mod_r, by = "Region")
    cmp_r = r_to_polars(cmp_r)
    compare_r_to_py(cmp_r, cmp_py)


def test_comparisons_by_true():
    cmp_py = comparisons(mod_py, by = True)
    cmp_r = marginaleffects.comparisons(mod_r, by = True)
    cmp_r = r_to_polars(cmp_r)
    compare_r_to_py(cmp_r, cmp_py)


def test_comparisons_by_false():
    cmp_py = comparisons(mod_py, by = False)
    cmp_r = marginaleffects.comparisons(mod_r, by = False)
    cmp_r = r_to_polars(cmp_r)
    compare_r_to_py(cmp_r, cmp_py)