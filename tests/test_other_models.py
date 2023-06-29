from marginaleffects.testing import *
from marginaleffects import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
from rpy2.robjects.packages import importr
from marginaleffects.comparisons import estimands
from pytest import approx
import patsy

importr("marginaleffects")
importr("nnet")


def test_smf_mixedlm():
    marginaleffects = importr("marginaleffects")
    lme4 = importr("lme4")
    dat_py, dat_r = rdatasets("geepack", "dietox", r = True)
    mod_py = smf.mixedlm("Weight ~ Time", dat_py, groups=dat_py["Pig"]).fit()
    mod_py = comparisons(mod_py, by = True)
    mod_r = lme4.lmer("Weight ~ Time + (1 | Pig)", dat_r)
    mod_r = marginaleffects.comparisons(mod_r, by = True)
    mod_r = r_to_polars(mod_r)
    assert mod_r["estimate"].to_numpy() == approx(mod_py["estimate"].to_numpy())


def test_mnlogit():
    importr("nnet")
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
    mod_r = nnet.multinom("island ~ bill_length_mm + flipper_length_mm", dat_r, trace = True)
    mod_py = smf.mnlogit("island ~ bill_length_mm + flipper_length_mm", dat_py).fit()
    pre_py = predictions(mod_py)
    pre_r = marginaleffects.predictions(mod_r)
    compare(pre_py, pre_r)