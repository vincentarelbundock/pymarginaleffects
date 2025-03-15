import sys
import pytest
from marginaleffects import *
from marginaleffects.plot_slopes import *
from tests.utilities import *
from tests.helpers import *

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="Skipped on macOS")

FIGURES_FOLDER = "plot_slopes"


@pytest.mark.plot
class TestPlotSlopes:
    def test_by(self, penguins_mod_add):
        fig = plot_slopes(penguins_mod_add, variables="species", by="island")
        assert assert_image(fig, "by_01", FIGURES_FOLDER) is None

        # fig = plot_slopes(mod, variables="bill_length_mm", by=["species", "island"])
        # assert assert_image(fig, "by_02", FIGURES_FOLDER) is None

    def test_condition(self, penguins_mod_add):
        fig = plot_slopes(
            penguins_mod_add,
            variables="bill_length_mm",
            condition=["flipper_length_mm", "species"],
            eps_vcov=1e-2,
        )
        assert assert_image(fig, "condition_01", FIGURES_FOLDER) is None

        fig = plot_slopes(
            penguins_mod_add, variables="species", condition="bill_length_mm"
        )
        assert assert_image(fig, "condition_02", FIGURES_FOLDER) is None

        fig = plot_slopes(
            penguins_mod_add, variables="island", condition="bill_length_mm", eps=1e-2
        )
        assert assert_image(fig, "condition_03", FIGURES_FOLDER) is None

        fig = plot_slopes(
            penguins_mod_add,
            variables="species",
            condition=["bill_length_mm", "species", "island"],
        )
        assert assert_image(fig, "condition_04", FIGURES_FOLDER) is None

    @pytest.mark.parametrize(
        "input_condition, input_variables, expected_figure_filename",
        [
            (
                {"wt": None, "cyl": None, "disp": None, "qsec": None},
                "hp",
                "issue114_slopes_01",
            ),
            (
                {"wt": None, "cyl": None, "disp": None, "qsec": None},
                "cyl",
                "issue114_slopes_02",
            ),
            (
                {"cyl": None, "wt": "minmax", "disp": None, "qsec": None},
                "hp",
                "issue114_slopes_03",
            ),
        ],
    )
    def test_issue114_slopes(
        self, input_condition, input_variables, expected_figure_filename, mtcars_mod
    ):
        fig = plot_slopes(
            mtcars_mod, variables=input_variables, condition=input_condition
        )
        assert assert_image(fig, expected_figure_filename, FIGURES_FOLDER) is None

    @pytest.mark.parametrize(
        "input_condition, input_variables, expected_figure_filename",
        [
            (
                {"flipper_length_mm": None, "species": ["Adelie", "Chinstrap"]},
                ["species"],
                "issue_114_03",
            ),
            (
                {"bill_length_mm": None, "flipper_length_mm": "minmax"},
                ["island"],
                "issue_114_04",
            ),
            (
                {"flipper_length_mm": None, "bill_length_mm": "fivenum"},
                ["island"],
                "issue_114_05",
            ),
            (
                {"flipper_length_mm": None, "bill_length_mm": "threenum"},
                ["island"],
                "issue_114_06",
            ),
            (
                {
                    "flipper_length_mm": None,
                    "bill_length_mm": None,
                    "island": None,
                },
                ["island"],
                "issue_114_07",
            ),
            (
                {
                    "flipper_length_mm": None,
                    "bill_length_mm": None,
                    "island": None,
                },
                ["species"],
                "issue_114_08",
            ),
        ],
    )
    def test_issue_114(
        self,
        input_condition,
        input_variables,
        expected_figure_filename,
        penguins_mod_add,
    ):
        fig = plot_slopes(
            penguins_mod_add, variables=input_variables, condition=input_condition
        )
        assert assert_image(fig, expected_figure_filename, FIGURES_FOLDER) is None
