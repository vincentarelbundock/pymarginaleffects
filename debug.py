import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from marginaleffects.testing import *
from rpy2.robjects.packages import importr

marginaleffects = importr("marginaleffects")
stats = importr("stats")
df_py, df_r = rdatasets("datasets", "mtcars", r = True)
mod_py = smf.ols("mpg ~ wt * hp", df_py).fit()
mod_py = comparisons(mod_py, comparison = "dyex").sort(["term", "contrast", "rowid"])
mod_r = stats.lm("mpg ~ wt * hp", data = df_r)
mod_r = marginaleffects.comparisons(mod_r, comparison = "dyex", data = df_r)
mod_r = r_to_polars(mod_r).sort(["term", "contrast", "rowid"])

compare_r_to_py(mod_r, mod_py, tolr = 2e-2)


print()
print(comparisons(mod, comparison = "dyex"))