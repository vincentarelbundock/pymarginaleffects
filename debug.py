import re
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from marginaleffects.testing import rdatasets
from marginaleffects.estimands import estimands
# Guerry
df, df_r = rdatasets("HistData", "Guerry", r = True)
df = df \
    .with_columns(
        (pl.col("Area") > pl.col("Area").median()).alias("Bool"),
        (pl.col("Distance") > pl.col("Distance").median()).alias("Bin")) \
    .with_columns(
        pl.col("Bin").apply(lambda x: int(x), return_dtype=pl.Int32).alias('Bin'),
        pl.Series(np.random.choice(["a", "b", "c"], df.shape[0])).alias("Char"))

fit = smf.ols("Literacy ~ Pop1831 * Desertion + Bool + Bin + Char", df).fit()
print(comparisons(fit, by = True))
# print(comparisons(fit, variables = "Pop1831", comparison = "differenceavg"))
# print(comparisons(fit, variables = "Pop1831", comparison = "difference").head())
# print(comparisons(fit, variables = "Pop1831", comparison = "ratio").head())
# print(comparisons(fit, variables = "Pop1831", comparison = "difference", by = "Region"))
# print(comparisons(fit, vcov = False, comparison = "differenceavg"))
# print(comparisons(fit, vcov = "HC3", comparison = "differenceavg"))
# print(comparisons(fit))
# print(comparisons(fit, variables = {"Char": "sequential"}))
# print(comparisons(fit, variables = "Pop1831"))
# print(comparisons(fit, variables = ["Pop1831", "Desertion"]))
# print(comparisons(fit, variables = {"Pop1831": 1000, "Desertion": 2}))
# print(comparisons(fit, variables = {"Pop1831": [100, 2000]}))


# # mtcars
# df = pl.read_csv("mtcars.csv")
# mod = smf.ols("mpg ~ wt * hp * cyl", df).fit()
# print(predictions(mod, by = True))
# print(predictions(mod, by = "cyl"))
# print(predictions(mod, by = ["cyl", "gear"]))
# print(predictions(mod, by = False))
# # print(comparisons(mod, comparison = "differenceavg"))
# # print(comparisons(mod, comparison = "difference", by = True))
# # print(comparisons(mod, comparison = "difference", by = False))
# # print(comparisons(mod, comparison = "difference", by = "gear"))