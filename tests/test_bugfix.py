import numpy as np
import polars as pl
import pandas as pd
from marginaleffects import predictions
import statsmodels.formula.api as smf


def test_issue_25():
    d = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
    train = d.head(50)
    test = d.tail(50)
    m = smf.ols('A ~ B + C + D', data=train).fit()
    p = predictions(m, newdata=test)
    assert isinstance(p, pl.DataFrame)
