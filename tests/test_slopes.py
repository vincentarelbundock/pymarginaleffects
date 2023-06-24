import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from pytest import approx
from marginaleffects import *
from marginaleffects.testing import *
from rpy2.robjects.packages import importr

# R packages
marginaleffects = importr("marginaleffects")
stats = importr("stats")

# # Guerry Data
# df = sm.datasets.get_rdataset("Guerry", "HistData").data
# df_r = pandas_to_r(df)
# df = pl.from_pandas(df)
# df = df.with_columns((pl.col("Area") > pl.col("Area").median()).alias("Bool"))
# df = df.with_columns((pl.col("Distance") > pl.col("Distance").median()).alias("Bin"))
# df = df.with_columns(df['Bin'].apply(lambda x: int(x), return_dtype=pl.Int32).alias('Bin'))
# df = df.with_columns(pl.Series(np.random.choice(["a", "b", "c"], df.shape[0])).alias("Char"))

# mod_py = smf.ols("Literacy ~ Pop1831 * Desertion", df).fit()
# mod_r = stats.lm("Literacy ~ Pop1831 * Desertion", data = df_r)
# slo_py = comparisons(mod_py, comparison = "dydx")
# slo_r = marginaleffects.comparisons(mod_r, comparison = "dydx", eps = 1e-4)
# slo_r = r_to_polars(slo_r)

# mod_py = smf.ols("Literacy ~ Pop1831 * Desertion", df).fit()
# mod_r = stats.lm("Literacy ~ Pop1831 * Desertion", data = df_r)
# slo_py = comparisons(mod_py, comparison = "dydxavg")
# slo_r = marginaleffects.comparisons(mod_r, comparison = "dydxavg", eps = 1e-4)
# slo_r = r_to_polars(slo_r)

# def test_basic():
#     cmp_py = comparisons(mod_py, comparison = "differenceavg")
#     cmp_r = marginaleffects.comparisons(mod_r, comparison = "differenceavg")
#     cmp_r = r_to_polars(cmp_r)
#     cmp_r = cmp_r.sort(["term", "contrast"])
#     cmp_py = cmp_r.sort(["term", "contrast"])
#     for col in ["estimate", "std_error", "statistic", "conf_low", "conf_high"]:
#         if col in cmp_py.columns and col in cmp_r.columns:
#             assert cmp_r[col].to_numpy() == approx(cmp_py[col].to_numpy(), rel = 1e-5)