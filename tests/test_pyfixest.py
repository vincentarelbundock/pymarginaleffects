import polars as pl
from polars.testing import assert_series_equal
import numpy as np
from pyfixest.estimation import feols
from pyfixest.utils import get_data
from marginaleffects import *


def test_bare_minimum():
    np.random.seed(1024)
    data = pl.DataFrame({
        "X1": np.random.normal(size = 1000),
        "X2": np.random.normal(size = 1000),
        "Z1": np.random.normal(size = 1000),
        "e": np.random.normal(size = 1000),
    }).with_columns((pl.col("X1") * pl.col("X2") * pl.col("Z1") + pl.col("e")).alias("Y"))
    fit = feols("Y ~ X1 * X2 * Z1", data = data.to_pandas())

    p = predictions(fit)
    assert p.shape == (1000, 13)

    p = avg_predictions(fit)
    assert_series_equal(p["estimate"], pl.Series([0.010447]),
                        check_names=False, rtol=1e-3)
    s = avg_slopes(fit)
    known = pl.Series([0.010960664156094414, -0.02592049598947146, 0.08384415120847774])
    assert_series_equal(s["estimate"], known, check_names=False, rtol=1e-4)

    c = comparisons(fit, newdata = datagrid(X1 = [2, 4], model = fit))
    known = pl.Series([0.025874865365717037, 0.02587486536571701, 0.06166037023566736, 0.14618691715896967, -0.12440034801695467, -0.2807823993887409])
    assert_series_equal(c["estimate"], known, check_names=False, rtol=1e-4)



# dat = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Titanic.csv").to_pandas()
# mod = feols("Survived ~ SexCode * Age * PClass", data = dat)
# avg_slopes(mod)
# plot_slopes(mod, variables = "SexCode", condition = ["Age", "PClass"])

# # average effect of an increase of 25 in age on the predicted probability of survival
# mod = feols("Survived ~ SexCode * Age | PClass", data = dat)
# avg_comparisons(mod, variables = {"Age": 50})