import patsy
import numpy as np
import pandas as pd
import polars as pl

import statsmodels.formula.api as smf
import statsmodels.api as sm
from marginaleffects import *
np.random.seed(1024)

# recode
df = sm.datasets.get_rdataset("Guerry", "HistData").data
df = pl.from_pandas(df)
df = df.with_columns(
        (pl.col("Area") > pl.col("Area").median()).alias("Bool"),
        (pl.col("Distance") > pl.col("Distance").median()).alias("Bin"))
df = df.with_columns(
        pl.col('Bin').apply(lambda x: int(x), return_dtype=pl.Int32).alias('Bin'),
        pl.Series(np.random.choice(["a", "b", "c"], df.shape[0])).alias("Char"))


# fit
mod = smf.ols("Literacy ~ Pop1831 * Desertion + Bool + Bin + Char", df)
# mod = smf.ols("Literacy ~ Pop1831 * Desertion + Bool + Bin", df)
fit = mod.fit()

print(hypotheses(fit, hypothesis = np.array([1, -1, 0, 0, 0, 0, 0, 0])))

print(hypotheses(fit, hypothesis = "b1 = b2 * 3"))

# print(comparisons(fit, comparison = "difference", vcov = "HC3"))
print(comparisons(fit, comparison = "dyexavg"))

# print(comparisons(fit, variables = {"Char": "pairwise"}, comparison = "differenceavg", vcov = "HC3"))


# hyp = hypotheses(fit, hypothesis = np.array([1, -1, 0, 0, 0, 0, 0, 0]))
# print(hyp)

# comparisons(fit, variables = {"Pop1831": 100, "Desertion": [0, 3]})





# comparisons(fit, variables = ["Pop1831", "Desertion"])
# comparisons(fit, variables = {"Desertion": 100})

# p = predictions(fit, by = "Region")
# p = predictions(fit, by = "Region", hypothesis = "reference", vcov = False)
# p = predictions(fit, by = "Region", hypothesis = "pairwise", vcov = False)

# hyp = np.vstack([
#     [1, 0, -1, 0, 0, 0],
#     [1, 0, 0, -1, 0, 0]
# ]).T
# predictions(fit, by = "Region", hypothesis = hyp)


# # predictions(fit, newdata = pl.from_pandas(df).head(), hypothesis = np.array(range(5)))

# # comparisons(fit, "Pop1831", value = 1, comparison = "differenceavg")

# # comparisons(fit, "Pop1831", value = 1, comparison = "difference")

# # # TODO: estimates work but not standard errors
# # comparisons(fit, "Pop1831", value = 1, comparison = "difference", by = "Region")
# # predictions(fit, by = "Region")


# # # Hypothesis
# df = sm.datasets.get_rdataset("Guerry", "HistData").data
# mod = smf.ols("Literacy ~ Pop1831 * Desertion", df)
# fit = mod.fit()
# df["bin"] = df["Literacy"] > df["Literacy"].median()
# df["bin"] = df["bin"].replace({True: 1, False: 0})
# mod = smf.glm("bin ~ Pop1831 * Desertion", df, family = sm.families.Binomial())
# fit = mod.fit()
# comparisons(fit)