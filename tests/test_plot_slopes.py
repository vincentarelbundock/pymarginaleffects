import polars as pl
import pytest
import statsmodels.formula.api as smf

from marginaleffects import *
from marginaleffects.plot_slopes import *

from .utilities import *

df = pl.read_csv(
    "https://vincentarelbundock.github.io/Rdatasets/csv/palmerpenguins/penguins.csv",
    null_values="NA",
).drop_nulls()
mod = smf.ols(
    "body_mass_g ~ flipper_length_mm * species * bill_length_mm * island",
    df.to_pandas(),
).fit()


def test_by():
    fig = plot_slopes(mod, variables="species", by="island")
    assert assert_image(fig, "by_01", "plot_slopes") is None

    fig = plot_slopes(mod, variables="bill_length_mm", by=["species", "island"])
    assert assert_image(fig, "by_02", "plot_slopes") is None


def test_condition():
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
