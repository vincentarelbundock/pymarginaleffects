# Linear model

``` python
import numpy as np
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
from marginaleffects import comparisons, predictions

df = sm.datasets.get_rdataset("Guerry", "HistData").data
df = pl.from_pandas(df)
df = df.with_columns((pl.col("Area") > pl.col("Area").median()).alias("Area_Bin"))
mod = smf.ols("Literacy ~ Pop1831 * Desertion + Area_Bin", df)
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

    /home/vincent/repos/pymarginaleffects/marginaleffects/comparisons.py:125: UserWarning: vcov is set to False because `by` or `hypothesis` is not None
      warn("vcov is set to False because `by` or `hypothesis` is not None")

<small>shape: (1, 7)</small>

| term      | contrast     | estimate  | std_error | statistic | conf_low   | conf_high |
|-----------|--------------|-----------|-----------|-----------|------------|-----------|
| str       | str          | f64       | f64       | f64       | f64        | f64       |
| "Pop1831" | "2000 - 100" | 14.610059 | 21.165112 | 0.69029   | -27.501875 | 56.721992 |


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

    11.388981576579823
    10.550624104384866

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

<small>shape: (2, 7)</small>

| term        | contrast | estimate | std_error | statistic | conf_low | conf_high |
|-------------|----------|----------|-----------|-----------|----------|-----------|
| str         | str      | f64      | f64       | f64       | f64      | f64       |
| "Pop1831"   | "+1"     | 0.000483 | 0.000318  | 1.520951  | -0.00014 | 0.001106  |
| "Desertion" | "+1"     | 0.005819 | 0.001642  | 3.544691  | 0.002601 | 0.009036  |

