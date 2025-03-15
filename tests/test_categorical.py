from marginaleffects import *
import polars as pl
import statsmodels.formula.api as smf
from polars.testing import assert_series_equal


hiv = get_dataset("thornton").to_pandas()
mtcars = get_dataset("mtcars", "datasets").to_pandas()


def test_avg_predictions_by_cat():
    # R results
    e = pl.Series([0.669811320754608, 0.672553348049938, 0.720383275261309])
    s = pl.Series([0.0241453345825489, 0.011617570113057, 0.0121533676309027])

    mod = smf.logit("outcome ~ agecat + incentive", hiv).fit()
    p = predictions(mod, by="agecat")
    assert_series_equal(p["estimate"], e, check_names=False)
    assert_series_equal(p["std_error"], s, check_names=False)
    k = (
        pl.DataFrame(hiv)
        .drop_nulls(subset=["outcome", "agecat", "incentive"])
        .to_pandas()
    )
    p = predictions(mod, by="agecat", newdata=k)
    assert_series_equal(p["estimate"], e, check_names=False)
    assert_series_equal(p["std_error"], s, check_names=False)


def test_mtcars_avg_slopes():
    mod = smf.ols("mpg ~ wt + C(gear)", data=mtcars).fit()
    s = avg_slopes(mod)
    assert s.shape[0] == 3
    assert all(s["contrast"] == ["4 - 3", "5 - 3", "dY/dX"])


def test_hiv_avg_slopes():
    mod = smf.ols("outcome ~ incentive + agecat", data=hiv).fit()
    s = avg_slopes(mod)
    assert s.shape[0] == 3
    assert all(s["contrast"] == ["18 to 35 - <18", ">35 - <18", "dY/dX"])
