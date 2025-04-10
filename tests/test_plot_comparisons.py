import sys
import pytest
from marginaleffects import *
from marginaleffects.plot_comparisons import *
from tests.utilities import *
from tests.helpers import *

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="Skipped on macOS")


FIGURES_FOLDER = "plot_comparisons"


@pytest.mark.plot
class TestPlotComparisons:
    def test_continuous(self, penguins_mod_add):
        fig = plot_comparisons(
            penguins_mod_add,
            variables="bill_length_mm",
            by="island",
        )
        assert assert_image(fig, "continuous_01", FIGURES_FOLDER) is None

        fig = plot_comparisons(
            penguins_mod_add,
            variables="bill_length_mm",
            condition=["flipper_length_mm", "species"],
        )
        assert assert_image(fig, "continuous_02", FIGURES_FOLDER) is None

        fig = plot_comparisons(
            penguins_mod_add, variables="bill_length_mm", condition="species"
        )
        assert assert_image(fig, "continuous_03", FIGURES_FOLDER) is None

    def test_discrete(self, penguins_mod_add):
        fig = plot_comparisons(
            penguins_mod_add,
            variables="species",
            condition=["bill_length_mm", "island"],
        )
        assert assert_image(fig, "discrete_01", FIGURES_FOLDER) is None

        fig = plot_comparisons(penguins_mod_add, variables="species", by="island")
        assert assert_image(fig, "discrete_02", FIGURES_FOLDER) is None

        fig = plot_comparisons(
            penguins_mod_add, variables="species", condition="bill_length_mm"
        )
        assert assert_image(fig, "discrete_03", FIGURES_FOLDER) is None

    @pytest.mark.parametrize(
        "variables, condition, gray, expected_file",
        [
            (
                "species",
                {
                    "bill_length_mm": None,
                    "island": None,
                    "flipper_length_mm": "threenum",
                },
                True,
                "threenum_horiz_axis_gray",
            ),
            (
                "species",
                {
                    "bill_length_mm": None,
                    "flipper_length_mm": "threenum",
                    "island": None,
                },
                True,
                "threenum_color_gray",
            ),
            (
                "species",
                {
                    "bill_length_mm": None,
                    "island": None,
                    "flipper_length_mm": "threenum",
                },
                False,
                "threenum_horiz_axis",
            ),
            (
                "species",
                {
                    "bill_length_mm": None,
                    "flipper_length_mm": "threenum",
                    "island": None,
                },
                False,
                "threenum_color",
            ),
        ],
    )
    def test_threenum(
        self, variables, condition, gray, expected_file, penguins_mod_add
    ):
        fig = plot_comparisons(
            penguins_mod_add, variables=variables, condition=condition, gray=gray
        )
        assert assert_image(fig, expected_file, FIGURES_FOLDER) is None
