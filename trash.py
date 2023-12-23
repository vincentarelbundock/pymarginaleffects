import polars as pl
import statsmodels.formula.api as smf

dat = pl.read_csv("~/repos/mfxplainer/data/impartiality.csv") \
  .with_columns(
    pl.col("impartial").cast(pl.Int8),
    pl.col(["democracy", "continent"]).cast(pl.Categorical),
  )

m = smf.logit(
  "impartial ~ equal * democracy + continent",
  data = dat.to_pandas()
).fit()

# from marginaleffects.model_abstract import *

from marginaleffects import *
k = ModelStatsmodels(m)

predictions(m)

k.modeldata