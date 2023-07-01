import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *

dat = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv")

mod = smf.ols("mpg ~ wt * qsec + disp", data = dat.to_pandas()).fit()
print(avg_comparisons(mod))