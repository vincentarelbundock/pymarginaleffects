import statsmodels.api as sm
import statsmodels.formula.api as smf
from pytest import approx
from marginaleffects import *
from marginaleffects.testing import *
from rpy2.robjects.packages import importr

# R packages
marginaleffects = importr("marginaleffects")
stats = importr("stats")

# Guerry Data
df = sm.datasets.get_rdataset("Guerry", "HistData").data
df_r = pandas_to_r(df)

def test_basic():
    mod_py = smf.ols("Literacy ~ Pop1831 * Desertion", df).fit()
    mod_r = stats.lm("Literacy ~ Pop1831 * Desertion", data = df_r)
    cmp_py = comparisons(mod_py, comparison = "differenceavg")
    cmp_r = marginaleffects.comparisons(mod_r, comparison = "differenceavg")
    cmp_r = r_to_polars(cmp_r)
    cmp_r = cmp_r.sort(["term", "contrast"])
    cmp_py = cmp_r.sort(["term", "contrast"])
    for col in ["estimate", "std_error", "statistic", "conf_low", "conf_high"]:
        if col in cmp_py.columns and col in cmp_r.columns:
            assert cmp_r[col].to_numpy() == approx(cmp_py[col].to_numpy(), rel = 1e-5)

def test_HC3():
    mod_py = smf.ols("Literacy ~ Pop1831 * Desertion", df).fit()
    mod_r = stats.lm("Literacy ~ Pop1831 * Desertion", data = df_r)
    cmp_py = comparisons(mod_py, comparison = "differenceavg", vcov = "HC3")
    cmp_r = marginaleffects.comparisons(mod_r, comparison = "differenceavg", vcov = "HC3")
    cmp_r = r_to_polars(cmp_r)
    cmp_r = cmp_r.sort(["term", "contrast"])
    cmp_py = cmp_r.sort(["term", "contrast"])
    for col in ["estimate", "std_error", "statistic", "conf_low", "conf_high"]:
        if col in cmp_py.columns and col in cmp_r.columns:
            assert cmp_r[col].to_numpy() == approx(cmp_py[col].to_numpy(), rel = 1e-5)