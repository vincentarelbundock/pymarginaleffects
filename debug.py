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


print(comparisons(mod_py, comparison = "difference", by = True))
# print(comparisons(mod_py, comparison = "difference", by = True))
# print(comparisons(mod_py, comparison = "ratio", by = True))