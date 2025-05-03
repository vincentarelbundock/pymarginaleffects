import re

import numpy as np
import polars as pl
import statsmodels.formula.api as smf
import statsmodels.api as sm
from polars.testing import (
    assert_series_equal,
    assert_frame_equal,
    assert_frame_not_equal,
)
import pytest

import marginaleffects
from marginaleffects import *
from marginaleffects.comparisons import estimands
from tests.helpers import mtcars, guerry

dat = guerry.with_columns(
    (pl.col("Area") > pl.col("Area").median()).alias("Boolea"),
    (pl.col("Distance") > pl.col("Distance").median()).alias("Bin"),
)
dat = dat.with_columns(
    pl.col("Bin").cast(pl.Int32),
    pl.Series(np.random.choice(["a", "b", "c"], dat.shape[0])).alias("Char"),
).to_pandas()

mod = smf.ols("Literacy ~ Pop1831 * Desertion", dat).fit()


def test_comparisons_hypothesis_list_length_3():
    dat = get_dataset("interaction_04")
    mod = smf.logit("Y ~ X * M1 * M2", data=dat).fit()
    comps_1 = avg_comparisons(
        mod, hypothesis="b2-b0=-4", variables="X", by=["M2", "M1"]
    )
    comps_2 = avg_comparisons(
        mod, hypothesis="b2-b1=-10", variables="X", by=["M2", "M1"]
    )
    comps_3 = avg_comparisons(mod, hypothesis="b2+b1=7", variables="X", by=["M2", "M1"])
    comps_three = avg_comparisons(
        mod,
        hypothesis=["b2-b0=-4", "b2-b1=-10", "b2+b1=7"],
        variables="X",
        by=["M2", "M1"],
    )
    assert_frame_equal(
        comps_three, pl.concat([comps_1, comps_2, comps_3], how="vertical")
    )
    assert_frame_not_equal(
        comps_three, pl.concat([comps_3, comps_2, comps_1], how="vertical")
    )


def test_comparisons_hypothesis_list():
    dat = get_dataset("interaction_04")
    mod = smf.logit("Y ~ X * M1 * M2", data=dat).fit()
    comps_1 = avg_comparisons(mod, hypothesis="b2=1", variables="X", by=["M2", "M1"])
    comps_2 = avg_comparisons(
        mod, hypothesis="b1 - b0 = 0", variables="X", by=["M2", "M1"]
    )

    comps_both = avg_comparisons(
        mod, hypothesis=["b1 - b0 = 0", "b2=1"], variables="X", by=["M2", "M1"]
    )

    assert_frame_not_equal(comps_both, pl.concat([comps_1, comps_2], how="vertical"))
    assert_frame_equal(comps_both, pl.concat([comps_2, comps_1], how="vertical"))


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
    assert isinstance(
        comparisons(fit), marginaleffects.classes.MarginaleffectsDataFrame
    )
    assert isinstance(
        comparisons(fit, variables="Pop1831", comparison="differenceavg"),
        marginaleffects.classes.MarginaleffectsDataFrame,
    )
    assert isinstance(
        comparisons(fit, variables="Pop1831", comparison="difference").head(),
        marginaleffects.classes.MarginaleffectsDataFrame,
    )
    assert isinstance(
        comparisons(fit, variables="Pop1831", comparison="ratio").head(),
        marginaleffects.classes.MarginaleffectsDataFrame,
    )
    assert isinstance(
        comparisons(fit, variables="Pop1831", comparison="difference", by="Region"),
        marginaleffects.classes.MarginaleffectsDataFrame,
    )
    assert isinstance(
        comparisons(fit, vcov=False, comparison="differenceavg"),
        marginaleffects.classes.MarginaleffectsDataFrame,
    )
    assert isinstance(
        comparisons(fit, vcov="HC3", comparison="differenceavg"),
        marginaleffects.classes.MarginaleffectsDataFrame,
    )
    assert isinstance(
        comparisons(fit), marginaleffects.classes.MarginaleffectsDataFrame
    )
    assert isinstance(
        comparisons(fit, variables={"Char": "sequential"}),
        marginaleffects.classes.MarginaleffectsDataFrame,
    )
    assert isinstance(
        comparisons(fit, variables="Pop1831"),
        marginaleffects.classes.MarginaleffectsDataFrame,
    )
    assert isinstance(
        comparisons(fit, variables=["Pop1831", "Desertion"]),
        marginaleffects.classes.MarginaleffectsDataFrame,
    )
    assert isinstance(
        comparisons(fit, variables={"Pop1831": 1000, "Desertion": 2}),
        marginaleffects.classes.MarginaleffectsDataFrame,
    )
    assert isinstance(
        comparisons(fit, variables={"Pop1831": [100, 2000]}),
        marginaleffects.classes.MarginaleffectsDataFrame,
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
