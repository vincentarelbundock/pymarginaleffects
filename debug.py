import re
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from marginaleffects.testing import rdatasets
from marginaleffects.estimands import estimands

# mtcars
df = pl.read_csv("mtcars.csv")
mod = smf.ols("mpg ~ wt * hp * cyl", df).fit()

def fun(x):
    return np.exp(x) 

out = comparisons(mod, transform = fun)
