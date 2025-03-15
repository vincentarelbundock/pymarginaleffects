import polars as pl
from polars.testing import assert_series_equal
import statsmodels.formula.api as smf
from tests.helpers import guerry, penguins, diamonds, mtcars
import marginaleffects
from marginaleffects import *
from tests.utilities import *

df = guerry.with_columns(pl.Series(range(guerry.shape[0])).alias("row_id")).sort(
    "Region", "row_id"
)
mod_py = smf.ols("Literacy ~ Pop1831 * Desertion", df).fit()


def test_newdata_balanced():
    mod = smf.ols(
        "body_mass_g ~ flipper_length_mm * species * bill_length_mm + island",
        penguins.to_pandas(),
    ).fit()
    p = predictions(mod, newdata="balanced")
    assert p.shape[0] == 9


def test_predictions():
    pre_py = predictions(mod_py)
    pre_r = pl.read_csv("tests/r/test_predictions_01.csv")
    compare_r_to_py(pre_r, pre_py)


def test_by():
    pre_py = predictions(mod_py, by="Region")
    pre_r = pl.read_csv("tests/r/test_predictions_02.csv")
    compare_r_to_py(pre_r, pre_py)


def test_by_2():
    pre_py = predictions(mod_py, by=["Region", "MainCity"])
    assert pre_py.shape == (13, 9)


def test_by_hypothesis():
    pre_py = predictions(mod_py, by="Region")
    pre_py = predictions(mod_py, by="Region", hypothesis="b0 * b2 = b2*2")
    pre_r = pl.read_csv("tests/r/test_predictions_03.csv")
    compare_r_to_py(pre_r, pre_py)


def test_class_manipulation():
    p = predictions(mod_py)
    assert isinstance(p, pl.DataFrame)
    assert isinstance(p, marginaleffects.classes.MarginaleffectsDataFrame)
    p = p.head()
    assert isinstance(p, pl.DataFrame)
    assert isinstance(p, marginaleffects.classes.MarginaleffectsDataFrame)


def issue_38():
    p = avg_predictions(mod_py, by=True)
    assert p.shape[0] == 1
    p = avg_predictions(mod_py)
    assert p.shape[0] == 1


def issue_59():
    p = predictions(mod_py, vcov=False)
    assert p.shape[0] == df.shape[0]
    assert p.shape[1] > 20


def test_issue_83():
    diamonds83 = diamonds.with_columns(
        cut_ideal_null=pl.when(pl.col("cut") == "Ideal")
        .then(pl.lit(None))
        .otherwise(pl.col("cut"))
    )

    model = smf.ols("price ~ cut_ideal_null", diamonds83.to_pandas()).fit()

    newdata = diamonds.slice(0, 20)
    newdata = newdata.with_columns(
        cut_ideal_null=pl.when(pl.col("cut") == "Ideal")
        .then(pl.lit("Premium"))
        .otherwise(pl.col("cut"))
    )

    p = predictions(model, newdata=newdata)
    assert p.shape[0] == newdata.shape[0]


def test_issue_95():
    model = smf.ols("price ~ cut + clarity + color", diamonds.to_pandas()).fit()

    newdata = diamonds.slice(0, 20)
    p = predictions(model, newdata=newdata, by="cut")

    newdata = newdata.with_columns(pred=pl.Series(model.predict(newdata.to_pandas())))
    newdata = newdata.group_by("cut").agg(pl.col("pred").mean())
    p = p.sort(by="cut")
    newdata = newdata.sort(by="cut")

    assert_series_equal(p["estimate"], newdata["pred"], check_names=False)


def test_issue161():
    mod = smf.ols("mpg ~ wt * hp * cyl", mtcars.to_pandas()).fit()

    # predictions(variables=)
    p = predictions(mod)
    assert p.shape[0] == 32
    p = predictions(mod, variables="cyl")
    assert p.shape[0] == 96
    p = predictions(mod, variables={"cyl": None})
    assert p.shape[0] == 96
    p = predictions(mod, variables=["cyl"])
    assert p.shape[0] == 96
    p = predictions(mod, variables=["cyl", "am"])
    assert p.shape[0] == 192

    # avg_predictions(variables=)
    p1 = avg_predictions(mod)
    p2 = avg_predictions(mod, variables="cyl")
    p3 = predictions(mod, variables="cyl")
    assert p2["estimate"][0] != p1["estimate"][0]
    assert p2["estimate"][0] == p3["estimate"].mean()
