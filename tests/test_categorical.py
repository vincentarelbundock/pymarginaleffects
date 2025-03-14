from marginaleffects import *
import polars as pl
import statsmodels.formula.api as smf
from polars.testing import assert_series_equal

dat = get_dataset("thornton")  # .with_columns(pl.col("agecat").cast(pl.Utf8))


def test_avg_predictions_by_cat():
    # R results
    e = pl.Series([0.669811320754608, 0.672553348049938, 0.720383275261309])
    s = pl.Series([0.0241453345825489, 0.011617570113057, 0.0121533676309027])

    mod = smf.logit("outcome ~ agecat + incentive", dat.to_pandas()).fit()
    p = predictions(mod, by="agecat")
    assert_series_equal(p["estimate"], e, check_names=False)
    assert_series_equal(p["std_error"], s, check_names=False)

    p = predictions(
        mod,
        by="agecat",
        newdata=dat.drop_nulls(subset=["agecat", "outcome", "incentive"]),
    )
    assert_series_equal(p["estimate"], e, check_names=False)
    assert_series_equal(p["std_error"], s, check_names=False)
