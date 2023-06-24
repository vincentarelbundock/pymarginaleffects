# Linear model

``` python
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
import statsmodels.api as sm
from marginaleffects import *

# load data
df = sm.datasets.get_rdataset("Guerry", "HistData").data
df = pl.from_pandas(df)

# boolean and binary recodes
df = df.with_columns((pl.col("Area") > pl.col("Area").median()).alias("Bool"))
df = df.with_columns((pl.col("Distance") > pl.col("Distance").median()).alias("Bin"))
df = df.with_columns(df['Bin'].apply(lambda x: int(x), return_dtype=pl.Int32).alias('Bin'))

# fit model
mod = smf.ols("Literacy ~ Pop1831 * Desertion + Bool + Bin + Region", df)
fit = mod.fit()
```

# Comparisons

``` python
# `comparison`
comparisons(fit)
comparisons(fit, variables = "Pop1831", comparison = "differenceavg")
comparisons(fit, variables = "Pop1831", comparison = "difference").head()
comparisons(fit, variables = "Pop1831", comparison = "ratio").head()

# `by`
comparisons(fit, variables = "Pop1831", comparison = "difference", by = "Region")

# `variables` argument
comparisons(fit)
comparisons(fit, variables = "Pop1831")
comparisons(fit, variables = ["Pop1831", "Desertion"])
comparisons(fit, variables = {"Pop1831": 1000, "Desertion": 2})
comparisons(fit, variables = {"Pop1831": [100, 2000]})
```

# `predictions()`

``` python
predictions(fit).head()
predictions(fit, by = "Region")

# `hypothesis` lincome matrices
hyp = np.array([1, 0, -1, 0, 0, 0])
predictions(fit, by = "Region", hypothesis = hyp)

hyp = np.vstack([
    [1, 0, -1, 0, 0, 0],
    [1, 0, 0, -1, 0, 0]
]).T
predictions(fit, by = "Region", hypothesis = hyp)

# equivalent to:
p = predictions(fit, by = "Region")
print(p["estimate"][0] - p["estimate"][2])
print(p["estimate"][0] - p["estimate"][3])
```

# GLM

``` python
df = sm.datasets.get_rdataset("Guerry", "HistData").data
mod = smf.ols("Literacy ~ Pop1831 * Desertion", df)
fit = mod.fit()
df["bin"] = df["Literacy"] > df["Literacy"].median()
df["bin"] = df["bin"].replace({True: 1, False: 0})
mod = smf.glm("bin ~ Pop1831 * Desertion", df, family = sm.families.Binomial())
fit = mod.fit()
comparisons(fit, comparison = "differenceavg")
```
