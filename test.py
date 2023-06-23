import numpy as np
import pandas as pd
import polars as pl # I know these transformations are costly, but I hate pandas and am trying to learn polars
import statsmodels.formula.api as smf
import statsmodels.api as sm
from marginaleffects import comparisons
df = sm.datasets.get_rdataset("Guerry", "HistData").data
mod = smf.ols("Literacy ~ Pop1831 * Desertion", df)
fit = mod.fit()

comparisons(fit, "Pop1831", value = 1, comparison = "differenceavg")

comparisons(fit, "Pop1831", value = 1, comparison = "difference")

comparisons(fit, "Pop1831", value = 1, comparison = "difference", by = "Region")


# estimand = np.random.normal(size = df.shape[0])

# get_by(fit, estimand = estimand, df = df, by = "Region")