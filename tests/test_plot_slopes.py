import polars as pl
import pytest
import statsmodels.formula.api as smf

from marginaleffects import *
from marginaleffects.plot_slopes import *

from .utilities import *


@pytest.fixture
def mod():
    df = pl.read_csv(
        "https://vincentarelbundock.github.io/Rdatasets/csv/palmerpenguins/penguins.csv",
        null_values="NA",
    ).drop_nulls()
    mod = smf.ols(
        "body_mass_g ~ flipper_length_mm * species * bill_length_mm * island",
        df.to_pandas(),
    ).fit()
    return mod

@pytest.fixture
def mtcars_mod():
    mtcars = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv")
    mod = smf.ols("mpg ~ hp * wt * disp * cyl", data = mtcars).fit()
    return mod

def test_by(mod):
    fig = plot_slopes(mod, variables="species", by="island")
    assert assert_image(fig, "by_01", "plot_slopes") is None

    fig = plot_slopes(mod, variables="bill_length_mm", by=["species", "island"])
    assert assert_image(fig, "by_02", "plot_slopes") is None

def test_condition(mod):
    fig = plot_slopes(
        mod,
        variables="bill_length_mm",
        condition=["flipper_length_mm", "species"],
        eps_vcov=1e-2,
    )
    assert assert_image(fig, "condition_01", "plot_slopes") is None

    fig = plot_slopes(mod, variables="species", condition="bill_length_mm")
    assert assert_image(fig, "condition_02", "plot_slopes") is None

    fig = plot_slopes(mod, variables="island", condition="bill_length_mm", eps=1e-2)
    assert assert_image(fig, "condition_03", "plot_slopes") is None

    fig = plot_slopes(
        mod, variables="species", condition=["bill_length_mm", "species", "island"]
    )
    assert assert_image(fig, "condition_04", "plot_slopes") is None

def test_issue114_slopes(mtcars_mod):
    fig = plot_slopes(mtcars_mod,
        variables = "hp",
        condition = {"wt": None,"cyl": None,"disp": None,"mpg":None,}
    )
    assert assert_image(fig, "issue114_slopes_01", "plot_slopes") is None
    # fig.save(r"C:\Users\amatvei\Projects\!work\vab\pymarginaleffects\fig.png")

@pytest.mark.parametrize(
    "input_condition, expected_figure_filename",
    [
        (["flipper_length_mm", "species"], "issue_114_01"),
        (["flipper_length_mm", "bill_length_mm"], "issue_114_02"),
        (
            {"flipper_length_mm": None, "species": ["Adelie", "Chinstrap"]},
            "issue_114_03",
        ),
        # ({"flipper_length_mm": None, "bill_length_mm": "threenum"}, "issue_114_04"),
        # ({"flipper_length_mm": None, "bill_length_mm": "fivenum"}, "issue_114_05"),
        # ({"flipper_length_mm": None, "bill_length_mm": "minmax"}, "issue_114_06"),
        (
            {
                "flipper_length_mm": None,
                "species": ["Adelie", "Chinstrap"],
                "bill_length_mm": None,
            },
            "issue_114_07",
        ),
        (
            {
                "flipper_length_mm": None,
                "species": ["Adelie", "Chinstrap"],
                "bill_length_mm": None,
                "island": None,
            },
            "issue_114_08",
        ),
    ],
)
def test_issue_114(input_condition, expected_figure_filename, mod):
    fig = plot_slopes(mod, condition=input_condition)
    assert assert_image(fig, expected_figure_filename, "plot_slopes") is None

