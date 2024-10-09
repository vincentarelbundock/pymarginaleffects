import sys
import polars as pl
import statsmodels.formula.api as smf
import pytest
from marginaleffects import *
from marginaleffects.plot_comparisons import *
from .utilities import *

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="Skipped on macOS")

FIGURES_FOLDER = "plot_comparisons"


def test_continuous(mod):
    fig = plot_comparisons(
        mod,
        variables="bill_length_mm",
        by="island",
    )
    assert assert_image(fig, "continuous_01", FIGURES_FOLDER) is None

    fig = plot_comparisons(
        mod, variables="bill_length_mm", condition=["flipper_length_mm", "species"]
    )
    assert assert_image(fig, "continuous_02", FIGURES_FOLDER) is None

    fig = plot_comparisons(mod, variables="bill_length_mm", condition="species")
    assert assert_image(fig, "continuous_03", FIGURES_FOLDER) is None


def test_discrete(mod):
    fig = plot_comparisons(
        mod, variables="species", condition=["bill_length_mm", "island"]
    )
    assert assert_image(fig, "discrete_01", FIGURES_FOLDER) is None

    fig = plot_comparisons(mod, variables="species", by="island")
    assert assert_image(fig, "discrete_02", FIGURES_FOLDER) is None

    fig = plot_comparisons(mod, variables="species", condition="bill_length_mm")
    assert assert_image(fig, "discrete_03", FIGURES_FOLDER) is None


@pytest.mark.parametrize(
    "variables, condition, expected_file",
    [
        (
            "species",
            {"bill_length_mm": None, "island": None, "flipper_length_mm": "threenum"},
            "threenum_horiz_axis",
        ),
        (
            "species",
            {"bill_length_mm": None, "flipper_length_mm": "threenum", "island": None},
            "threenum_color",
        ),
    ],
)
def test_threenum(variables, condition, expected_file, mod):
    fig = plot_comparisons(mod, variables=variables, condition=condition)
    assert assert_image(fig, expected_file, FIGURES_FOLDER) is None
