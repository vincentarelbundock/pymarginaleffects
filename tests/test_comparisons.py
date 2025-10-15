import re

import numpy as np
import polars as pl
import statsmodels.formula.api as smf
import statsmodels.api as sm
from polars.testing import assert_series_equal
import pytest

from marginaleffects import *
from marginaleffects.comparisons import estimands
from tests.helpers import mtcars, guerry

hiv = get_dataset("thornton")
dat = guerry.with_columns(
    (pl.col("Area") > pl.col("Area").median()).alias("Boolea"),
    (pl.col("Distance") > pl.col("Distance").median()).alias("Bin"),
)
dat = dat.with_columns(
    pl.col("Bin").cast(pl.Int32),
    pl.Series(np.random.choice(["a", "b", "c"], dat.shape[0])).alias("Char"),
).to_pandas()

mod = smf.ols("Literacy ~ Pop1831 * Desertion", dat).fit()


def test_difference():
    cmp_py = comparisons(mod, comparison="differenceavg").sort("term")
    cmp_r = pl.read_csv("tests/r/test_comparisons_01.csv").sort("term")
    assert_series_equal(cmp_py["estimate"], cmp_r["estimate"])
    assert_series_equal(
        cmp_py["std_error"], cmp_r["std.error"], check_names=False, rtol=1e-3
    )
    cmp_py = comparisons(mod, comparison="difference").sort("term", "rowid")
    cmp_r = pl.read_csv("tests/r/test_comparisons_02.csv").sort("term", "rowid")
    assert_series_equal(cmp_py["estimate"], cmp_r["estimate"])
    assert_series_equal(
        cmp_py["std_error"], cmp_r["std.error"], check_names=False, rtol=1e-3
    )


def test_comparison_simple():
    est = [k for k in estimands.keys() if not re.search("x|wts", k)]
    for e in est:
        cmp_py = comparisons(mod, comparison=e).sort("term")
        cmp_r = pl.read_csv(f"tests/r/test_comparisons_03_{e}.csv").sort("term")
        if cmp_r.shape[1] == 170:
            raise ValueError("R and Python results are not the same")
        assert_series_equal(cmp_py["estimate"], cmp_r["estimate"], rtol=1e-2)
        assert_series_equal(
            cmp_py["std_error"], cmp_r["std.error"], check_names=False, rtol=3e-2
        )


def test_by():
    cmp_py = comparisons(mod, comparison="differenceavg", by="Region").sort(
        "term", "Region"
    )
    cmp_r = pl.read_csv("tests/r/test_comparisons_04.csv").sort("term", "Region")
    assert_series_equal(cmp_py["estimate"], cmp_r["estimate"])
    assert_series_equal(
        cmp_py["std_error"], cmp_r["std.error"], check_names=False, rtol=1e-3
    )


def test_HC3():
    cmp_py = comparisons(mod, comparison="differenceavg", vcov="HC3").sort("term")
    cmp_r = pl.read_csv("tests/r/test_comparisons_05.csv").sort("term")
    assert_series_equal(cmp_py["estimate"], cmp_r["estimate"])
    assert_series_equal(
        cmp_py["std_error"], cmp_r["std.error"], check_names=False, rtol=1e-3
    )


@pytest.mark.parametrize("vcov_str", ["HC0", "HC1", "HC2", "HC3"])
def test_vcov(vcov_str):
    cmp_py = comparisons(mod, comparison="differenceavg", vcov=vcov_str).sort("term")
    assert cmp_py.shape == (2, 9)


def test_difference_wts():
    cmp_py = comparisons(mod, variables="Desertion", by="Region", wts="Literacy")
    cmp_r = pl.read_csv("tests/r/test_comparisons_06.csv").sort("Region")
    assert_series_equal(cmp_py["estimate"], cmp_r["estimate"])
    assert_series_equal(
        cmp_py["std_error"], cmp_r["std.error"], check_names=False, rtol=1e-4
    )
    cmp_py = comparisons(mod, variables="Desertion", by="Region")
    cmp_r = pl.read_csv("tests/r/test_comparisons_07.csv").sort("Region")
    assert_series_equal(cmp_py["estimate"], cmp_r["estimate"])
    assert_series_equal(
        cmp_py["std_error"], cmp_r["std.error"], check_names=False, rtol=1e-4
    )


def test_bare_minimum():
    fit = smf.ols(
        "Literacy ~ Pop1831 * Desertion + Boolea + Bin + Char", data=dat
    ).fit()
    res = comparisons(fit)
    assert isinstance(res, MarginaleffectsResult)
    assert isinstance(res.data, pl.DataFrame)

    assert isinstance(
        comparisons(fit, variables="Pop1831", comparison="differenceavg"),
        MarginaleffectsResult,
    )
    diff_result = comparisons(fit, variables="Pop1831", comparison="difference")
    assert isinstance(diff_result, MarginaleffectsResult)
    assert isinstance(diff_result.head(), pl.DataFrame)
    ratio_result = comparisons(fit, variables="Pop1831", comparison="ratio")
    assert isinstance(ratio_result, MarginaleffectsResult)
    assert isinstance(ratio_result.head(), pl.DataFrame)
    assert isinstance(
        comparisons(fit, variables="Pop1831", comparison="difference", by="Region"),
        MarginaleffectsResult,
    )
    assert isinstance(
        comparisons(fit, vcov=False, comparison="differenceavg"),
        MarginaleffectsResult,
    )
    assert isinstance(
        comparisons(fit, vcov="HC3", comparison="differenceavg"),
        MarginaleffectsResult,
    )
    assert isinstance(
        comparisons(fit, variables={"Char": "sequential"}),
        MarginaleffectsResult,
    )
    assert isinstance(
        comparisons(fit, variables="Pop1831"),
        MarginaleffectsResult,
    )
    assert isinstance(
        comparisons(fit, variables=["Pop1831", "Desertion"]),
        MarginaleffectsResult,
    )
    assert isinstance(
        comparisons(fit, variables=["Pop1831", "Desertion"], by=True),
        MarginaleffectsResult,
    )
    assert isinstance(
        comparisons(
            fit,
            variables=["Pop1831", "Desertion"],
            comparison="ratio",
            hypothesis="Pop1831 = Desertion",
        ),
        MarginaleffectsResult,
    )
    assert isinstance(
        comparisons(fit, comparison="difference", hypothesis=None),
        MarginaleffectsResult,
    )
    assert isinstance(
        comparisons(fit, comparison="difference", hypothesis="Pop1831 = 0"),
        MarginaleffectsResult,
    )


def test_variables_function():
    def forward_diff(x):
        return pl.DataFrame({"base": x, "forward": x + 10})

    def backward_diff(x):
        return pl.DataFrame({"backward": x - 10, "base": x})

    def center_diff(x):
        return pl.DataFrame({"low": x - 5, "high": x + 5})

    mod = smf.glm("vs ~ hp", data=mtcars, family=sm.families.Binomial()).fit()

    cmp_py = comparisons(mod, variables={"hp": forward_diff})
    cmp_r = pl.read_csv("tests/r/test_comparisons_08_forward_diff.csv")
    assert_series_equal(cmp_py["estimate"], cmp_r["estimate"])
    assert_series_equal(cmp_py["std_error"], cmp_r["std.error"], check_names=False)
    cmp_py = comparisons(mod, variables={"hp": backward_diff})
    cmp_r = pl.read_csv("tests/r/test_comparisons_08_backward_diff.csv")
    assert_series_equal(cmp_py["estimate"], cmp_r["estimate"])
    assert_series_equal(cmp_py["std_error"], cmp_r["std.error"], check_names=False)
    cmp_py = comparisons(mod, variables={"hp": center_diff})
    cmp_r = pl.read_csv("tests/r/test_comparisons_08_center_diff.csv")
    assert_series_equal(cmp_py["estimate"], cmp_r["estimate"])
    assert_series_equal(cmp_py["std_error"], cmp_r["std.error"], check_names=False)


def test_contrast():
    mod = smf.ols("mpg ~ hp * qsec", data=mtcars).fit()
    comp = avg_comparisons(mod, variables={"hp": "2sd"})
    assert comp["contrast"].item(), "mean((x+sd)) - mean((x-sd))"


def test_lift():
    mod = smf.ols("am ~ hp", data=mtcars).fit()
    cmp1 = comparisons(mod, comparison="lift")
    cmp2 = comparisons(mod, comparison="liftavg")
    assert cmp1.shape[0] == 32
    assert cmp2.shape[0] == 1
    with pytest.raises(AssertionError):
        comparisons(mod, comparison="liftr")


def test_issue192():
    mod = smf.logit(
        "outcome ~ incentive * (agecat + distance)", data=hiv.to_pandas()
    ).fit()
    grid = pl.DataFrame({"distance": 2, "agecat": ["18 to 35"], "incentive": 1})
    cmp = comparisons(mod, variables="incentive", newdata=grid)
    print(cmp)


def test_issue193():
    mod = smf.logit(
        "outcome ~ incentive * (agecat + distance)", data=hiv.to_pandas()
    ).fit()
    grid = pl.DataFrame({"distance": 2, "agecat": ["18 to 35"], "incentive": 1})
    g_treatment = grid.with_columns(pl.lit(1).alias("incentive"))
    g_control = grid.with_columns(pl.lit(0).alias("incentive"))
    p_treatment = mod.predict(g_treatment.to_pandas())
    p_control = mod.predict(g_control.to_pandas())
    cmp1 = (p_treatment - p_control).to_list()

    # This should work with and without specifying Enum levels
    cmp2 = comparisons(mod, variables="incentive", newdata=grid, vcov=False)
    assert cmp2["estimate"].is_in(cmp1).all()

    # This should work with and without specifying Enum levels
    grid = grid.with_columns(pl.col("agecat").cast(pl.Enum(["<18", "18 to 35", ">35"])))
    cmp2 = comparisons(mod, variables="incentive", newdata=grid, vcov=False)
    assert cmp2["estimate"].is_in(cmp1).all()


def test_issue197():
    mod = smf.logit(
        "outcome ~ incentive * (agecat + distance)", data=hiv.to_pandas()
    ).fit()
    grid = pl.DataFrame({"distance": 2, "agecat": ["18 to 35"], "incentive": 1})
    cmp = comparisons(mod, variables={"incentive": [1, 0]}, newdata=grid)
    assert (cmp["estimate"] < 0).all()


def test_issue198():
    mod = smf.logit(
        "outcome ~ incentive * (agecat + distance)", data=hiv.to_pandas()
    ).fit()
    grid = pl.DataFrame({"distance": 2, "agecat": ["18 to 35"], "incentive": 1})
    cmp = comparisons(mod, variables={"distance": "iqr"}, newdata=grid)
    assert (cmp["std_error"] > 0).all()
    assert (cmp["estimate"] < 0).all()


def test_issue82_cross():
    mtcars = get_dataset("mtcars", "datasets").to_pandas()
    mod = smf.ols("mpg ~ am * hp", data=mtcars).fit()
    cmp = avg_comparisons(mod, variables=["am", "hp"], cross=True)
    np.testing.assert_almost_equal(
        cmp["estimate"].to_numpy(),
        np.array([5.21781687]),
    )
    assert cmp.height == 1
    cmp = comparisons(mod, variables=["am", "hp"], cross=True)
    assert cmp.height == 32


def test_issue_206():
    dat = get_dataset("thornton")
    dat = dat.to_pandas()
    mod = smf.logit("outcome ~ incentive * (agecat + distance)", data=dat).fit()
    cmp = avg_comparisons(
        mod, variables=["incentive", "distance"], cross=True, by="agecat"
    )
    assert cmp.height == 3
    assert (cmp["term"] == "cross").all()


def test_callable_comparison():
    """Test that comparison= argument accepts lambda functions to compute custom estimates.

    This test verifies that custom lambda functions produce identical results to
    built-in comparison methods (difference, ratio, lnratio).
    """
    mod = smf.ols("am ~ hp", data=mtcars).fit()

    # Test 1: Lambda difference should equal built-in difference
    cmp_builtin_diff = comparisons(
        mod, variables="hp", comparison="difference", vcov=False
    )
    cmp_lambda_diff = comparisons(
        mod, variables="hp", comparison=lambda hi, lo, eps, x, y, w: hi - lo, vcov=False
    )
    assert isinstance(cmp_lambda_diff, MarginaleffectsResult)
    np.testing.assert_array_almost_equal(
        cmp_builtin_diff["estimate"].to_numpy(), cmp_lambda_diff["estimate"].to_numpy()
    )

    # Test 2: Lambda ratio should equal built-in ratio
    cmp_builtin_ratio = comparisons(mod, variables="hp", comparison="ratio", vcov=False)
    cmp_lambda_ratio = comparisons(
        mod, variables="hp", comparison=lambda hi, lo, eps, x, y, w: hi / lo, vcov=False
    )
    assert isinstance(cmp_lambda_ratio, MarginaleffectsResult)
    assert all(cmp_lambda_ratio["estimate"] > 0)  # Ratio should be positive
    np.testing.assert_array_almost_equal(
        cmp_builtin_ratio["estimate"].to_numpy(),
        cmp_lambda_ratio["estimate"].to_numpy(),
    )

    # Test 3: Lambda lnratio should equal built-in lnratio
    cmp_builtin_lnratio = comparisons(
        mod, variables="hp", comparison="lnratio", vcov=False
    )
    cmp_lambda_lnratio = comparisons(
        mod,
        variables="hp",
        comparison=lambda hi, lo, eps, x, y, w: np.log(hi / lo),
        vcov=False,
    )
    assert isinstance(cmp_lambda_lnratio, MarginaleffectsResult)
    np.testing.assert_array_almost_equal(
        cmp_builtin_lnratio["estimate"].to_numpy(),
        cmp_lambda_lnratio["estimate"].to_numpy(),
    )

    # Test 4: Custom function (percent change) with avg_comparisons
    cmp_avg = avg_comparisons(
        mod,
        variables="hp",
        comparison=lambda hi, lo, eps, x, y, w: (hi - lo) / lo * 100,
        vcov=False,
        by=True,
    )
    assert isinstance(cmp_avg, MarginaleffectsResult)
    assert cmp_avg.shape[0] >= 1
