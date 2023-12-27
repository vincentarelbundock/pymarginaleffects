import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from marginaleffects.plot_predictions import *
from .utilities import *

penguins = pl.read_csv(
    "https://vincentarelbundock.github.io/Rdatasets/csv/palmerpenguins/penguins.csv",
    null_values="NA",
).drop_nulls()


def test_by():
    mod = smf.ols(
        "body_mass_g ~ flipper_length_mm * species * bill_length_mm + island",
        penguins.to_pandas(),
    ).fit()

    fig = plot_predictions(mod, by="species")
    assert assert_image(fig, "by_01", "plot_predictions") is None

    fig = plot_predictions(mod, by=["species", "island"])
    assert assert_image(fig, "by_02", "plot_predictions") is None


def test_condition():
    mod = smf.ols(
        "body_mass_g ~ flipper_length_mm * species * bill_length_mm + island",
        penguins.to_pandas(),
    ).fit()

    fig = plot_predictions(
        mod,
        condition={"flipper_length_mm": list(range(180, 220)), "species": None},
    )
    assert assert_image(fig, "condition_01", "plot_predictions") is None

    fig = plot_predictions(mod, condition=["bill_length_mm", "species", "island"])
    assert assert_image(fig, "condition_02", "plot_predictions") is None


def test_issue_57():
    mtcars = pl.read_csv(
        "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv"
    )
    mod = smf.ols("mpg ~ wt + am + qsec", mtcars.to_pandas()).fit()

    fig = plot_predictions(mod, condition=["qsec", "am"])
    assert assert_image(fig, "issue_57_01", "plot_predictions") is None

    fig = plot_predictions(
        mod,
        condition={
            "am": None,
            "qsec": [mtcars["qsec"].min(), mtcars["qsec"].max()],
        },
    )
    assert assert_image(fig, "issue_57_02", "plot_predictions") is None

    fig = plot_predictions(mod, condition={"wt": None, "am": None})
    assert assert_image(fig, "issue_57_03", "plot_predictions") is None

    fig = plot_predictions(mod, condition={"am": None, "wt": None})
    assert assert_image(fig, "issue_57_04", "plot_predictions") is None


def issue_62():
    import types

    mtcars = pl.read_csv(
        "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv"
    )
    mod = smf.ols("mpg ~ hp * wt * am", data=mtcars).fit()
    cond = {
        "hp": None,
        "wt": [
            mtcars["wt"].mean() - mtcars["wt"].std(),
            mtcars["wt"].mean(),
            mtcars["wt"].mean() + mtcars["wt"].std(),
        ],
        "am": None,
    }
    p = plot_predictions(mod, condition=cond)
    assert isinstance(p, types.ModuleType)
