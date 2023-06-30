import re
import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from marginaleffects.testing import rdatasets
from marginaleffects.estimands import estimands
from typing import Union
import patsy
import pandas as pd
import numpy as np


dat_py, dat_r = rdatasets("palmerpenguins", "penguins", r = True)
dat_py = dat_py \
    .with_columns(
        pl.col("island").cast(pl.Categorical),
        pl.col("bill_length_mm").map_dict({"NA": None}, default = pl.col("bill_length_mm")),
        pl.col("body_mass_g").map_dict({"NA": None}, default = pl.col("body_mass_g")),
        pl.col("flipper_length_mm").map_dict({"NA": None}, default = pl.col("flipper_length_mm")),) \
    .with_columns(
        pl.col("island").cast(pl.Int16),
        pl.col("bill_length_mm").cast(pl.Float32),
        pl.col("body_mass_g").cast(pl.Float32),
        pl.col("flipper_length_mm").cast(pl.Float32),
    )

# # dat_py = dat_py.drop_nulls()
# mod = smf.ols("body_mass_g ~ bill_length_mm + flipper_length_mm", dat_py).fit()
# print(avg_comparisons(mod, variables = {"flipper_length_mm": 100}, comparison = "lnor"))
# print(avg_predictions(mod))


dat_py = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv")

mod_py = smf.quantreg("mpg ~ wt + qsec", data = dat_py).fit(q = 0.5)
pre_py = predictions(mod)




dat = mtcars \
  .with_columns(
    pl.col("am").cast(pl.Boolean),
    pl.col("cyl").cast(pl.Utf8).cast(pl.Categorical),
  )
mod_cat = smf.ols("mpg ~ am + cyl + hp", data = dat).fit()
dat = mtcars \
  .with_columns(
    pl.col("am").cast(pl.Boolean),
    pl.col("cyl").cast(pl.Utf8).cast(pl.Categorical),
  )
mod = smf.ols("mpg ~ qsec * drat", data = mtcars).fit()
# mod.params
# mod_cat = smf.ols("mpg ~ am + cyl + hp", data = dat).fit()
# hypotheses(mod, "drat = 2 * qsec")

print(hypotheses(mod, "b4 - 2. * b3 = 0"))