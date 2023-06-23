import numpy as np
import pandas as pd
import polars as pl # I know these transformations are costly, but I hate pandas and am trying to learn polars
import statsmodels.formula.api as smf
import statsmodels.api as sm
from marginaleffects import *
df = sm.datasets.get_rdataset("Guerry", "HistData").data
mod = smf.ols("Literacy ~ Pop1831 * Desertion", df)
fit = mod.fit()


predictions(fit, by = "Region")
hyp = np.vstack([
    [1, 0, -1, 0, 0, 0],
    [1, 0, 0, -1, 0, 0]
]).T
predictions(fit, by = "Region", hypothesis = hyp)


# predictions(fit, newdata = pl.from_pandas(df).head(), hypothesis = np.array(range(5)))

# comparisons(fit, "Pop1831", value = 1, comparison = "differenceavg")

# comparisons(fit, "Pop1831", value = 1, comparison = "difference")

# # TODO: estimates work but not standard errors
# comparisons(fit, "Pop1831", value = 1, comparison = "difference", by = "Region")
# predictions(fit, by = "Region")

# # Hypothesis
