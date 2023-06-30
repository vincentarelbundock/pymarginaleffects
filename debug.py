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

# dat_py = dat_py.drop_nulls()
mod = smf.ols("body_mass_g ~ bill_length_mm + flipper_length_mm", dat_py).fit()
print(avg_comparisons(mod, comparison = "ratio"))
print(avg_predictions(mod))