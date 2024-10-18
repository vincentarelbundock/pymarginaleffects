import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import pytest
import marginaleffects
from marginaleffects import *


def test_error():
    N = 1000
    dat = pd.DataFrame({"x": np.random.normal(size=N)})
    dat["y"] = 1 + 2 * dat["x"] + np.random.normal(size=N)
    mod = sm.ols("y ~ x", data=dat).fit()
    s = avg_slopes(mod)
    assert isinstance(s, marginaleffects.classes.MarginaleffectsDataFrame)

    with pytest.raises(ValueError):
        mod = sm.ols("y ~ scale(x)", data=dat).fit()
        s = avg_slopes(mod)

    with pytest.raises(ValueError):
        mod = sm.ols("y ~ center(x)", data=dat).fit()
        s = avg_slopes(mod)
