import polars as pl
from marginaleffects import *
from polars.testing import assert_series_equal
from tests.conftest import wage_panel_pd
from linearmodels.panel import PanelOLS
from marginaleffects import fit_linearmodels

formula = "lwage ~ exper * hours * educ * married - 1"
data = wage_panel_pd
mod = fit_linearmodels(formula, data, engine=PanelOLS)


def test_predictions_01():
    unknown = predictions(mod)
    known = pl.read_csv("tests/r/test_linearmodels_panelols_predictions_01.csv")
    assert_series_equal(unknown["estimate"], known["estimate"])
    assert_series_equal(unknown["std_error"], known["std_error"], check_names=False)


def test_predictions_02():
    unknown = predictions(mod, by="married")
    known = pl.read_csv("tests/r/test_linearmodels_panelols_predictions_02.csv")
    assert_series_equal(unknown["estimate"], known["estimate"])
    assert_series_equal(unknown["std_error"], known["std_error"], check_names=False)


def test_comparisons_01():
    unknown = comparisons(mod).sort(["term", "contrast", "rowid"])
    known = pl.read_csv("tests/r/test_linearmodels_panelols_comparisons_01.csv").sort(
        ["term", "contrast", "rowid"]
    )
    assert_series_equal(unknown["estimate"], known["estimate"])
    assert_series_equal(unknown["std_error"], known["std_error"], check_names=False)


def test_comparisons_02():
    unknown = comparisons(mod, by="married").sort(["term", "married"])
    known = pl.read_csv("tests/r/test_linearmodels_panelols_comparisons_02.csv").sort(
        ["term", "married"]
    )
    assert_series_equal(unknown["estimate"], known["estimate"])
    assert_series_equal(unknown["std_error"], known["std_error"], check_names=False)


def test_slopes_01():
    unknown = slopes(mod)
    known = pl.read_csv("tests/r/test_linearmodels_panelols_slopes_01.csv")
    unknown = unknown.sort(["term", "contrast", "rowid"])
    known = known.sort(["term", "contrast", "rowid"])
    assert_series_equal(unknown["estimate"], known["estimate"], rtol=1e-4)
    # TODO: bad tolerance
    assert_series_equal(
        unknown["std_error"], known["std_error"], check_names=False, rtol=1e-1
    )


def test_slopes_02():
    unknown = slopes(mod, by="married").sort(["term", "married"])
    known = pl.read_csv("tests/r/test_linearmodels_panelols_slopes_02.csv").sort(
        ["term", "married"]
    )
    assert_series_equal(unknown["estimate"], known["estimate"])
    assert_series_equal(
        unknown["std_error"], known["std_error"], check_names=False, rtol=1e-2
    )
