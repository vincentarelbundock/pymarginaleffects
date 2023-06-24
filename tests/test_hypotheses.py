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
df = sm.datasets.get_rdataset("Guerry", "HistData").data
df_r = pandas_to_r(df)

# fit models
mod_py = smf.ols("Literacy ~ Pop1831 * Desertion", df).fit()
mod_r = stats.lm("Literacy ~ Pop1831 * Desertion", data = df_r)

def test_coefs():
    hyp_py = hypotheses(mod_py, hypothesis = np.array([1, -1, 0, 0]))
    hyp_r = marginaleffects.hypotheses(mod_r, hypothesis = "b1 - b2 = 0")
    hyp_r = r_to_polars(hyp_r)
    for col in ["estimate", "std_error", "statistic", "conf_low", "conf_high"]:
        if col in hyp_py.columns and col in hyp_r.columns:
            assert hyp_r[col].to_numpy() == approx(hyp_py[col].to_numpy(), rel = 1e-5)
