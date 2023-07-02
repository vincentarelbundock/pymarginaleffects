import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from scipy.stats import logistic

mtcars = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv")
mod = smf.logit("am ~ mpg", data = mtcars).fit()

mfx = slopes(mod, newdata = datagrid(mpg = 24, newdata = mtcars))
print(mfx)

beta_0 = mod.params.iloc[0]
beta_1 = mod.params.iloc[1]
print(beta_1 * logistic.pdf(beta_0 + beta_1 * 24))



print(avg_slopes(mod))


np.mean(beta_1 * logistic.pdf(beta_0 + beta_1 * mtcars["mpg"]))