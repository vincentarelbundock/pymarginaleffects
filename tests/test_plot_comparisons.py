import os

import polars as pl
import pytest
import statsmodels.formula.api as smf
from matplotlib.testing.compare import compare_images

from marginaleffects import *
from marginaleffects.plot_comparisons import *

from .utilities import *


def assert_image(fig, label, file, tolerance=5):
    known_path = f"./tests/images/{file}/"
    unknown_path = f"./tests/images/.tmp_{file}/"
    if os.path.isdir(unknown_path):
        for root, dirs, files in os.walk(unknown_path):
            for fname in files:
                os.remove(os.path.join(root, fname))
        os.rmdir(unknown_path)
    os.mkdir(unknown_path)
    unknown = f"{unknown_path}{label}.png"
    known = f"{known_path}{label}.png"
    if not os.path.exists(known):
        fig.savefig(known)
        raise FileExistsError(f"File {known} does not exist. Creating it now.")
    fig.savefig(unknown)
    out = compare_images(known, unknown, tol=tolerance)
    compare_images(known, unknown, tol=tolerance)
    os.remove(unknown)
    return out


df = pl.read_csv(
    "https://vincentarelbundock.github.io/Rdatasets/csv/palmerpenguins/penguins.csv",
    null_values="NA",
).drop_nulls()
mod = smf.ols(
    "body_mass_g ~ flipper_length_mm * species * bill_length_mm + island",
    df.to_pandas(),
).fit()


def test_plot_comparisons():
    fig = plot_comparisons(mod, variables="species", by="island")
    assert assert_image(fig, "Figure_1", "plot_comparisons") is None

    fig = plot_comparisons(
        mod,
        variables="bill_length_mm",
        by="island",
    )
    assert assert_image(fig, "Figure_2", "plot_comparisons") is None

    fig = plot_comparisons(
        mod, variables="bill_length_mm", condition=["flipper_length_mm", "species"]
    )
    assert assert_image(fig, "Figure_3", "plot_comparisons") is None

    fig = plot_comparisons(mod, variables="species", condition="bill_length_mm")
    assert assert_image(fig, "Figure_4", "plot_comparisons") is None

    fig = plot_comparisons(mod, variables="bill_length_mm", condition="species")
    assert assert_image(fig, "Figure_5", "plot_comparisons") is None

    fig = plot_comparisons(
        mod, variables="species", condition=["bill_length_mm", "species", "island"]
    )
    assert assert_image(fig, "Figure_6", "plot_comparisons") is None