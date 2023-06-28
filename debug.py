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


dat_py, dat_r = rdatasets("palmerpenguins", "penguins", r = True)
dat_py = dat_py \
    .with_columns(
        pl.col("island").cast(pl.Categorical),
        pl.col("bill_length_mm").map_dict({"NA": None}, default = pl.col("bill_length_mm")),
        pl.col("flipper_length_mm").map_dict({"NA": None}, default = pl.col("flipper_length_mm")),) \
    .with_columns(
        pl.col("island").cast(pl.Int16),
        pl.col("bill_length_mm").cast(pl.Float32),
        pl.col("flipper_length_mm").cast(pl.Float32),
    )

mod = smf.mnlogit("island ~ bill_length_mm + flipper_length_mm", dat_py).fit()

predictions(mod).filter(pl.col("rowid") == 2)