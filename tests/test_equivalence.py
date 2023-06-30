import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from pytest import approx
from marginaleffects import *
from .utilities import *
from rpy2.robjects.packages import importr
from marginaleffects.comparisons import estimands

marginaleffects = importr("marginaleffects")
stats = importr("stats")

df, df_r = rdatasets("HistData", "Guerry", r = True)
mod_py = smf.ols("Literacy ~ Pop1831 * Desertion", df).fit()
mod_r = stats.lm("Literacy ~ Pop1831 * Desertion", df_r)


def test_simple_equivalence():
    cmp_py = comparisons(mod_py, comparison = "differenceavg", equivalence = [-.1, .1])
    cmp_r = marginaleffects.comparisons(mod_r, comparison = "differenceavg", equivalence = .1)
    cmp_r = r_to_polars(cmp_r)
    cmp_r = cmp_r.sort("term")
    cmp_py = cmp_py.sort("term")
    assert cmp_r["statistic.nonsup"].to_numpy() == approx(cmp_py["statistic_nonsup"].to_numpy(), rel = 1e-3)
    assert cmp_r["p.value.nonsup"].to_numpy() == approx(cmp_py["p_value_nonsup"].to_numpy(), rel = 1e-3)
    cmp_r = marginaleffects.comparisons(mod_r, comparison = "differenceavg", equivalence = -.1)
    cmp_r = r_to_polars(cmp_r)
    cmp_r = cmp_r.sort("term")
    cmp_py = cmp_py.sort("term")
    assert cmp_r["statistic.noninf"].to_numpy() == approx(cmp_py["statistic_noninf"].to_numpy(), rel = 1e-3)
    assert cmp_r["p.value.noninf"].to_numpy() == approx(cmp_py["p_value_noninf"].to_numpy(), rel = 1e-3)
