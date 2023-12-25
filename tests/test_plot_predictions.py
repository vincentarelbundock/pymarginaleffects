import os

import polars as pl
import pytest
import statsmodels.formula.api as smf
from matplotlib.testing.compare import compare_images

from marginaleffects import *
from marginaleffects.plot_predictions import *

from .utilities import *


def test_image(fig, label, file, tolerance = 5):
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
    out = compare_images(known, unknown, tol = tolerance)
    compare_images(known, unknown, tol = tolerance)
    os.remove(unknown)
    return out




def test_basic():

    df = pl.read_csv(
        "https://vincentarelbundock.github.io/Rdatasets/csv/palmerpenguins/penguins.csv",
        null_values="NA",
    ).drop_nulls()
    mod = smf.ols(
        "body_mass_g ~ flipper_length_mm * species * bill_length_mm + island",
        df.to_pandas(),
    ).fit()

    fig = plot_predictions(mod, by="species")
    assert test_image(fig, "basic_01", "plot_predictions") is None

    fig = plot_predictions(
        mod, by="bill_length_mm", newdata=datagrid(model=mod, bill_length_mm=[37, 39])
    )
    assert test_image(fig, "basic_02", "plot_predictions") is None

    fig = plot_predictions(
        mod,
        by=["bill_length_mm", "island", "species"],
        newdata=datagrid(
            model=mod, bill_length_mm=[72, 431], species=["Adelie", "Chinstrap", "Gentoo"]
        ),
    )
    assert test_image(fig, "basic_03", "plot_predictions") is None

    fig = plot_predictions(mod, condition="bill_length_mm")
    assert test_image(fig, "basic_04", "plot_predictions") is None

    fig = plot_predictions(
        mod,
        condition={"flipper_length_mm": [i for i in range(180, 220)], "species": None},
    )
    assert test_image(fig, "basic_05", "plot_predictions") is None

    fig = plot_predictions(mod, condition=["bill_length_mm", "species", "island"])
    assert test_image(fig, "basic_06", "plot_predictions") is None




def test_issue_57():
    mtcars = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv")
    mod = smf.ols("mpg ~ wt + am + qsec", mtcars.to_pandas()).fit()

    fig = plot_predictions(mod, condition=["qsec", "am"])
    assert test_image(fig, "issue_57_01", "plot_predictions") is None

    fig = plot_predictions(mod, condition={
        "am": None,
        "qsec": [mtcars["qsec"].min(), mtcars["qsec"].max()],
        })
    assert test_image(fig, "issue_57_02", "plot_predictions") is None

    fig = plot_predictions(mod, condition={"wt": None, "am": None})
    assert test_image(fig, "issue_57_03", "plot_predictions") is None

    fig = plot_predictions(mod, condition={"am": None, "wt": None})
    assert test_image(fig, "issue_57_04", "plot_predictions") is None

