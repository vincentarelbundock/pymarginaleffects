import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *

df = pl.read_csv("mtcars.csv")
mod = smf.ols("mpg ~ wt * hp * cyl", df).fit()
# print(predictions(mod, by = ["cyl"]).head())
print(comparisons(mod, comparison = "difference").select(["term", "contrast", "estimate", "std_error"]))
# print(comparisons(mod, comparison = "difference", by = False))
# print(comparisons(mod, comparison = "difference", by = ["cyl", "gear"]))
# print(comparisons(mod_py, comparison = "difference", by = True))
# print(comparisons(mod_py, comparison = "ratio", by = True))