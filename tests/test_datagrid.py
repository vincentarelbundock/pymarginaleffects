import polars as pl

from marginaleffects import *
from tests.conftest import mtcars_df


def test_FUN_numeric():
    d = datagrid(newdata=mtcars_df, FUN_numeric=lambda x: x.median())
    assert d["cyl"][0] == mtcars_df["cyl"].median()
    assert d["hp"][0] == mtcars_df["hp"].median()
    assert d["carb"][0] == mtcars_df["carb"].median()


def test_simple_grid():
    d = datagrid(mpg=24, newdata=mtcars_df)
    assert d.shape == (1, 12)
    d = datagrid(mpg=[23, 24], hp=[120, 130], newdata=mtcars_df)
    assert d.shape == (4, 12)


def test_cf():
    assert datagrid(newdata=mtcars_df, mpg=32).shape[0] == 1
    assert (
        datagrid(newdata=mtcars_df, mpg=[30, 32], grid_type="counterfactual").shape[0]
        == 64
    )
    assert (
        datagrid(
            newdata=mtcars_df, mpg=32, am=0, hp=100, grid_type="counterfactual"
        ).shape[0]
        == 32
    )
    assert (
        datagrid(
            newdata=mtcars_df, am=[0, 1], hp=[100, 110, 120], grid_type="counterfactual"
        ).shape[0]
        == 192
    )
    assert (
        datagrid(newdata=mtcars_df, mpg=[30, 32], grid_type="counterfactual")
        .unique("rowidcf")
        .shape[0]
        == 32
    )
    assert set(
        datagrid(newdata=mtcars_df, mpg=[30, 32], grid_type="counterfactual").columns
    ) == {
        "gear",
        "qsec",
        "mpg",
        "cyl",
        "am",
        "wt",
        "vs",
        "drat",
        "rowidcf",
        "disp",
        "rownames",
        "hp",
        "carb",
    }
