import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from tests.utilities import *
import pytest
from tests.conftest import impartiality_df as dat

m = smf.logit("impartial ~ equal * democracy + continent", data=dat.to_pandas()).fit()


@pytest.mark.skip(reason="to be fixed")
def test_predictions():
    p = predictions(m)
    assert isinstance(p, pl.DataFrame)

    p = predictions(m, newdata=dat.head())
    assert isinstance(p, pl.DataFrame)

    p = predictions(m, newdata="mean")
    assert isinstance(p, pl.DataFrame)

    p = predictions(
        m,
        newdata=datagrid(model=m, democracy=dat["democracy"].unique(), equal=[30, 90]),
    )
    assert isinstance(p, pl.DataFrame)
    assert p.shape[0] == 4

    p1 = avg_predictions(m)
    p2 = np.mean(m.predict(dat.to_pandas()).to_numpy())
    assert isinstance(p1, pl.DataFrame)
    assert p1.shape[0] == 1
    assert p1["estimate"][0] == p2

    p = predictions(m, by="democracy")
    assert isinstance(p, pl.DataFrame)
    assert p.shape[0] == 2

    p = plot_predictions(m, by=["democracy", "continent"])
    assert assert_image(p, label="jss_01", file="jss") is None


def test_hypotheses():
    # hypotheses(m, hypothesis = "continentAsia = continentAmericas")
    h = hypotheses(m, hypothesis="b4 = b3")
    assert isinstance(h, pl.DataFrame)
    assert h.shape[0] == 1

    avg_predictions(m, by="democracy", hypothesis="revpairwise")

    p = predictions(m, by="democracy", hypothesis="b1 = b0 * 2")
    assert isinstance(p, pl.DataFrame)
    assert p.shape[0] == 1

    p = predictions(
        m, by="democracy", hypothesis="b1 = b0 * 2", equivalence=[-0.2, 0.2]
    )
    assert isinstance(p, pl.DataFrame)
    assert p.shape[0] == 1

    c = comparisons(m, variables="democracy")
    assert isinstance(c, pl.DataFrame)
    assert c.shape[0] == 166

    c = avg_comparisons(m)
    assert isinstance(c, pl.DataFrame)
    assert c.shape[0] == 5

    c = avg_comparisons(m, variables={"equal": 4})
    assert isinstance(c, pl.DataFrame)
    assert c.shape[0] == 1
    c = avg_comparisons(m, variables={"equal": "sd"})
    assert isinstance(c, pl.DataFrame)
    assert c.shape[0] == 1
    c = avg_comparisons(m, variables={"equal": [30, 90]})
    assert isinstance(c, pl.DataFrame)
    assert c.shape[0] == 1
    c = avg_comparisons(m, variables={"equal": "iqr"})
    assert isinstance(c, pl.DataFrame)
    assert c.shape[0] == 1

    c = avg_comparisons(m, variables="democracy", comparison="ratio")
    assert c["contrast"][0] == "mean(Democracy) / mean(Autocracy)"


def test_hypothesis_shape():
    m = smf.logit(
        "impartial ~ equal * democracy + continent", data=dat.to_pandas()
    ).fit()

    for h in [
        "reference",
        "revreference",
        "sequential",
        "revsequential",
        "pairwise",
        "revpairwise",
    ]:
        for b in ["democracy", "continent"]:
            c = comparisons(
                m,
                by=b,
                variables={"equal": [30, 90]},
                hypothesis=h,
            )
            assert isinstance(c, pl.DataFrame)
            if b == "democracy":
                assert c.shape[0] == 1
            else:
                assert c.shape[0] > 1


def test_transform():
    c1 = avg_comparisons(m, comparison="lnor")
    c2 = avg_comparisons(m, comparison="lnor", transform=np.exp)
    all(np.exp(c1["estimate"]) == c2["estimate"])


# avg_comparisons(m,
#   variables = "equal",
#   comparison = lambda hi, lo: np.mean(hi) / np.ean(lo))


def test_misc():
    cmp = comparisons(m, by="democracy", variables={"equal": [30, 90]})
    assert isinstance(cmp, pl.DataFrame)
    assert cmp.shape[0] == 2

    cmp = comparisons(
        m,
        by="democracy",
        variables={"equal": [30, 90]},
        hypothesis="pairwise",
    )
    assert isinstance(cmp, pl.DataFrame)
    assert cmp.shape[0] == 1

    s = slopes(m, variables="equal", newdata=datagrid(equal=[25, 50], model=m))
    assert isinstance(s, pl.DataFrame)
    assert s.shape[0] == 2

    s = avg_slopes(m, variables="equal")
    assert isinstance(s, pl.DataFrame)
    assert s.shape[0] == 1

    s = slopes(m, variables="equal", newdata="mean")
    assert isinstance(s, pl.DataFrame)
    assert s.shape[0] == 1

    s = avg_slopes(m, variables="equal", slope="eyex")
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


def test_python_section():
    p = avg_predictions(m, by="continent")
    assert isinstance(p, pl.DataFrame)
    assert p.shape[0] == 4

    s = slopes(m, newdata="mean")
    assert isinstance(s, pl.DataFrame)
    assert s.shape[0] == 5


def test_files_hosted_online():
    dat = pl.DataFrame("https://marginalffects.com/data/impartiality.csv")
    assert isinstance(dat, pl.DataFrame)
    dat = pl.DataFrame("https://marginalffects.com/data/titanic.csv")
    assert isinstance(dat, pl.DataFrame)
