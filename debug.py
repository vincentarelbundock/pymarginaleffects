import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
np.random.seed(1024)

# download and recode data
df = rdatasets("HistData", "Guerry") \
    .with_columns(
        (pl.col("Area") > pl.col("Area").median()).alias("Bool"),
        (pl.col("Distance") > pl.col("Distance").median()).alias("Bin")) \
    .with_columns(
        pl.col('Bin').apply(lambda x: int(x), return_dtype=pl.Int32).alias('Bin'),
        pl.Series(np.random.choice(["a", "b", "c"], df.shape[0])).alias("Char"))

# fit linear model with interaction
mod = smf.ols("Literacy ~ Pop1831 * Desertion + Bool + Bin * Char", df).fit()

print(hypotheses(fit, hypothesis = np.array([1, -1, 0, 0, 0, 0, 0, 0])))
print(hypotheses(fit, hypothesis = "b1 = b2 * 3"))
print(comparisons(fit, comparison = "dyexavg"))