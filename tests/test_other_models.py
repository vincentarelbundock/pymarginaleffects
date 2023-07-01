from .utilities import *
from marginaleffects import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
from rpy2.robjects.packages import importr
from marginaleffects.comparisons import estimands
from pytest import approx
import patsy

importr("marginaleffects")
importr("nnet")

mod_py = smf.quantreg("mpg ~ wt + qsec", data = dat_py).fit(q = 0.5)
pre_py = predictions(mod)

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


importr("quantreg")
dat_py, dat_r = rdatasets("datasets", "mtcars", r = True)
mod_py = smf.quantreg("mpg ~ wt + qsec", data = dat_py).fit(q = 0.25)
pre_py = predictions(mod_py)
cmp_py = comparisons(mod_py)
quantreg.rq("mpg ~ wt + qsec", data = dat_r, tau = 0.25)

