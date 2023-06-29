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
df = df.filter(pl.col("Region") != "NA")
mod_py = smf.ols("Literacy ~ Pop1831 * Desertion", df).fit()
mod_r = stats.lm("Literacy ~ Pop1831 * Desertion", data = df_r)


print(comparisons(mod_py, variables = "Desertion", by = "Region"))
print(comparisons(mod_py, variables = "Desertion", by = "Region", wts = "Crime_pers"))

