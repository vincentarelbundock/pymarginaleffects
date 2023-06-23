import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from marginaleffects import comparisons
df = sm.datasets.get_rdataset("Guerry", "HistData").data
mod = smf.ols("Literacy ~ Pop1831 * Desertion", df)
fit = mod.fit()

comparisons(fit, "Pop1831", value = 1, comparison = "differenceavg")

comparisons(fit, "Pop1831", value = 1, comparison = "difference").head()
