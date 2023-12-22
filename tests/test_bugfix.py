import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf

from marginaleffects import predictions


def test_issue_25():
    d = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))
    train = d.head(50)
    test = d.tail(50)
    m = smf.ols("A ~ B + C + D", data=train).fit()
    p = predictions(m, newdata=test)
    assert isinstance(p, pl.DataFrame)
