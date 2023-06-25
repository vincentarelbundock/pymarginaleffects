import numpy as np
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
df, df_r = rdatasets("HistData", "Guerry", r = True)

# fit models
mod_py = smf.ols("Literacy ~ Pop1831 * Desertion", df).fit()
mod_r = stats.lm("Literacy ~ Pop1831 * Desertion", data = df_r)

def test_coefs():
    hyp_py = hypotheses(mod_py, hypothesis = np.array([1, -1, 0, 0]))
    hyp_r = marginaleffects.hypotheses(mod_r, hypothesis = "b1 - b2 = 0")
    hyp_r = r_to_polars(hyp_r)
    compare_r_to_py(hyp_r, hyp_py)
