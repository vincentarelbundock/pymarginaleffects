import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from tests.utilities import *
import pytest
from tests.conftest import impartiality_df as dat


@pytest.mark.skip(reason="to be fixed")
def test_predictions(impartiality_model):
    p = predictions(impartiality_model)

    assert isinstance(p, pl.DataFrame)

    p = predictions(impartiality_model, newdata=dat.head())

    assert isinstance(p, pl.DataFrame)

    p = predictions(impartiality_model, newdata="mean")

    assert isinstance(p, pl.DataFrame)

    p = predictions(
        impartiality_model,
        newdata=datagrid(
            model=impartiality_model,
            democracy=dat["democracy"].unique(),
            equal=[30, 90],
        ),
    )
    assert isinstance(p, pl.DataFrame)
    assert p.shape[0] == 4

    p1 = avg_predictions(impartiality_model)
    p2 = np.mean(impartiality_model.predict(dat.to_pandas()).to_numpy())
    assert isinstance(p1, pl.DataFrame)
    assert p1.shape[0] == 1
    assert p1["estimate"][0] == p2

    p = predictions(impartiality_model, by="democracy")
    assert isinstance(p, pl.DataFrame)
    assert p.shape[0] == 2

    p = plot_predictions(impartiality_model, by=["democracy", "continent"])
    assert assert_image(p, label="jss_01", file="jss") is None


def test_hypotheses(impartiality_model):
    # hypotheses(m, hypothesis = "continentAsia = continentAmericas")

    h = hypotheses(impartiality_model, hypothesis="b4 = b3")

    assert isinstance(h, pl.DataFrame)
    assert h.shape[0] == 1

    avg_predictions(impartiality_model, by="democracy", hypothesis="revpairwise")

    p = predictions(impartiality_model, by="democracy", hypothesis="b1 = b0 * 2")
    assert isinstance(p, pl.DataFrame)
    assert p.shape[0] == 1

    p = predictions(
        impartiality_model,
        by="democracy",
        hypothesis="b1 = b0 * 2",
        equivalence=[-0.2, 0.2],
    )
    assert isinstance(p, pl.DataFrame)
    assert p.shape[0] == 1

    c = comparisons(impartiality_model, variables="democracy")
    assert isinstance(c, pl.DataFrame)
    assert c.shape[0] == 166

    c = avg_comparisons(impartiality_model)
    assert isinstance(c, pl.DataFrame)
    assert c.shape[0] == 5

    c = avg_comparisons(impartiality_model, variables={"equal": 4})
    assert isinstance(c, pl.DataFrame)
    assert c.shape[0] == 1
    c = avg_comparisons(impartiality_model, variables={"equal": "sd"})
    assert isinstance(c, pl.DataFrame)
    assert c.shape[0] == 1
    c = avg_comparisons(impartiality_model, variables={"equal": [30, 90]})
    assert isinstance(c, pl.DataFrame)
    assert c.shape[0] == 1
    c = avg_comparisons(impartiality_model, variables={"equal": "iqr"})
    assert isinstance(c, pl.DataFrame)
    assert c.shape[0] == 1

    c = avg_comparisons(impartiality_model, variables="democracy", comparison="ratio")

    assert c["contrast"][0] == "Democracy / Autocracy"

    c = avg_comparisons(impartiality_model, variables="democracy", comparison="differenceavg")
    assert c["contrast"][0] == "Democracy - Autocracy"



@pytest.mark.parametrize(
    "h, label",
    [
        (
            "reference",
            {
                "democracy": "Democracy - Autocracy",
                "continent": ["Americas - Africa", "Asia - Africa", "Europe - Africa"],
            },
        ),
        (
            "revreference",
            {
                "democracy": "Autocracy - Democracy",
                "continent": ["Africa - Americas", "Africa - Asia", "Africa - Europe"],
            },
        ),
        (
            "sequential",
            {
                "democracy": "Democracy - Autocracy",
                "continent": ["Americas - Africa", "Asia - Americas", "Europe - Asia"],
            },
        ),
        (
            "revsequential",
            {
                "democracy": "Autocracy - Democracy",
                "continent": ["Africa - Americas", "Americas - Asia", "Asia - Europe"],
            },
        ),
        (
            "pairwise",
            {
                "democracy": "Autocracy - Democracy",
                "continent": [
                    "Africa - Americas",
                    "Africa - Asia",
                    "Africa - Europe",
                    "Americas - Asia",
                    "Americas - Europe",
                    "Asia - Europe",
                ],
            },
        ),
        (
            "revpairwise",
            {
                "democracy": "Democracy - Autocracy",
                "continent": [
                    "Americas - Africa",
                    "Asia - Africa",
                    "Europe - Africa",
                    "Asia - Americas",
                    "Europe - Americas",
                    "Europe - Asia",
                ],
            },
        ),
    ],
)
def test_hypothesis_shape_and_row_labels(h, label, impartiality_model):
    for b in ["democracy", "continent"]:
        c = comparisons(
            impartiality_model,
            by=b,
            variables={"equal": [30, 90]},
            hypothesis=h,
        )
        assert isinstance(c, pl.DataFrame)
        if b == "democracy":
            assert c.shape[0] == 1
            assert c["term"][0] == label[b]
        else:
            assert c.shape[0] > 1
            assert (c["term"] == label[b]).all()


def test_transform(impartiality_model):
    c1 = avg_comparisons(impartiality_model, comparison="lnor")
    c2 = avg_comparisons(impartiality_model, comparison="lnor", transform=np.exp)
    all(np.exp(c1["estimate"]) == c2["estimate"])


# avg_comparisons(m,
#   variables = "equal",
#   comparison = lambda hi, lo: np.mean(hi) / np.ean(lo))


def test_misc(impartiality_model):
    cmp = comparisons(impartiality_model, by="democracy", variables={"equal": [30, 90]})
    assert isinstance(cmp, pl.DataFrame)
    assert cmp.shape[0] == 2

    cmp = comparisons(
        impartiality_model,
        by="democracy",
        variables={"equal": [30, 90]},
        hypothesis="pairwise",
    )
    assert isinstance(cmp, pl.DataFrame)
    assert cmp.shape[0] == 1

    s = slopes(
        impartiality_model,
        variables="equal",
        newdata=datagrid(equal=[25, 50], model=impartiality_model),
    )

    assert isinstance(s, pl.DataFrame)
    assert s.shape[0] == 2

    s = avg_slopes(impartiality_model, variables="equal")
    assert isinstance(s, pl.DataFrame)
    assert s.shape[0] == 1

    s = slopes(impartiality_model, variables="equal", newdata="mean")
    assert isinstance(s, pl.DataFrame)
    assert s.shape[0] == 1

    s = avg_slopes(impartiality_model, variables="equal", slope="eyex")
    assert isinstance(s, pl.DataFrame)
    assert s.shape[0] == 1


def test_titanic():
    tit = pl.read_csv("tests/data/titanic.csv")
    mod_tit = smf.ols("Survived ~ Woman * Passenger_Class", data=tit.to_pandas()).fit()

    p = avg_predictions(
        mod_tit,
        newdata=datagrid(
            Passenger_Class=tit["Passenger_Class"].unique(),
            Woman=tit["Woman"].unique(),
            model=mod_tit,
        ),
        by="Woman",
    )
    assert isinstance(p, pl.DataFrame)
    assert p.shape[0] == 2

    p = avg_predictions(
        mod_tit,
        newdata=datagrid(
            Passenger_Class=tit["Passenger_Class"].unique(),
            Woman=tit["Woman"].unique(),
            model=mod_tit,
        ),
        by="Woman",
        hypothesis="revpairwise",
    )
    assert isinstance(p, pl.DataFrame)
    assert p.shape[0] == 1

    p = avg_comparisons(
        mod_tit,
        variables="Woman",
        newdata=datagrid(
            Passenger_Class=tit["Passenger_Class"].unique(),
            Woman=tit["Woman"].unique(),
            model=mod_tit,
        ),
    )
    assert isinstance(p, pl.DataFrame)
    assert p.shape[0] == 1

    # Risk difference by passenger class
    c = avg_comparisons(
        mod_tit, variables="Woman", by="Passenger_Class", comparison="difference"
    )
    assert isinstance(c, pl.DataFrame)
    assert c.shape[0] == 3

    c = avg_comparisons(
        mod_tit, variables="Woman", by="Passenger_Class", hypothesis="b0 - b2 = 0"
    )
    assert isinstance(c, pl.DataFrame)
    assert c.shape[0] == 1


def test_python_section(impartiality_model):
    p = avg_predictions(impartiality_model, by="continent")
    assert isinstance(p, pl.DataFrame)
    assert p.shape[0] == 4

    s = slopes(impartiality_model, newdata="mean")
    assert isinstance(s, pl.DataFrame)
    assert s.shape[0] == 5


def test_files_hosted_online():
    dat = pl.DataFrame("https://marginalffects.com/data/impartiality.csv")
    assert isinstance(dat, pl.DataFrame)
    dat = pl.DataFrame("https://marginalffects.com/data/titanic.csv")
    assert isinstance(dat, pl.DataFrame)
