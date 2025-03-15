import numpy as np
import polars as pl

from marginaleffects import *
from statsmodels.formula.api import logit

mtcars = get_dataset("mtcars", "datasets")


def test_FUN_numeric():
    d = datagrid(newdata=mtcars, FUN_numeric=lambda x: x.median())
    assert d["cyl"][0] == mtcars["cyl"].median()
    assert d["hp"][0] == mtcars["hp"].median()
    assert d["carb"][0] == mtcars["carb"].median()


def test_simple_grid():
    d = datagrid(mpg=24, newdata=mtcars)
    assert d.shape == (1, 12)
    d = datagrid(mpg=[23, 24], hp=[120, 130], newdata=mtcars)
    assert d.shape == (4, 12)


def test_cf():
    assert datagrid(newdata=mtcars, mpg=32).shape[0] == 1
    assert (
        datagrid(newdata=mtcars, mpg=[30, 32], grid_type="counterfactual").shape[0]
        == 64
    )
    assert (
        datagrid(
            newdata=mtcars, mpg=32, am=0, hp=100, grid_type="counterfactual"
        ).shape[0]
        == 32
    )
    assert (
        datagrid(
            newdata=mtcars, am=[0, 1], hp=[100, 110, 120], grid_type="counterfactual"
        ).shape[0]
        == 192
    )
    assert (
        datagrid(newdata=mtcars, mpg=[30, 32], grid_type="counterfactual")
        .unique("rowidcf")
        .shape[0]
        == 32
    )
    assert set(
        datagrid(newdata=mtcars, mpg=[30, 32], grid_type="counterfactual").columns
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


def test_issue156():
    rng = np.random.default_rng(seed=48103)
    N = 10
    dat = pl.DataFrame(
        {
            "Num": rng.normal(loc=0, scale=1, size=N),
            "Bin": rng.binomial(n=1, p=0.5, size=N),
            "Cat": rng.choice(["A", "B", "C"], size=N),
        }
    )
    d = datagrid(grid_type="balanced", newdata=dat)
    assert d.shape[0] == 6
    assert set(d["Bin"].unique()) == {0, 1}
    assert set(d["Cat"].unique()) == {"A", "B", "C"}
    assert len(d["Num"].unique()) == 1


def test_mean_or_mode():
    d1 = datagrid(newdata=mtcars)
    d2 = datagrid(newdata=mtcars, grid_type="mean_or_mode")
    for col in d1.columns[1:]:
        assert (d1[col] == d2[col]).all()


def test_issue_169():
    dat = get_dataset("thornton")
    dat = dat.drop_nulls(subset=["incentive"])
    mod = logit("outcome ~ incentive + agecat + distance", data=dat.to_pandas()).fit()
    p = predictions(mod, newdata="balanced")
    assert p.shape[0] == 6
