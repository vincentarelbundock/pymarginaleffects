import polars as pl
import pytest
import statsmodels.formula.api as smf
from linearmodels.datasets import wage_panel


diamonds = pl.read_csv("tests/data/diamonds.csv")

dietox = pl.read_csv("tests/data/dietox.csv")

guerry = pl.read_csv(
    "tests/data/Guerry.csv",
    null_values="NA",
).drop_nulls()

guerry_with_nulls = pl.read_csv("tests/data/Guerry.csv")

impartiality_df = pl.read_csv("tests/data/impartiality.csv").with_columns(
    pl.col("impartial").cast(pl.Int8)
)

iris = pl.read_csv("tests/data/iris.csv")

mtcars = pl.read_csv("tests/data/mtcars.csv")

penguins = pl.read_csv(
    "tests/data/penguins.csv",
    null_values="NA",
).drop_nulls()

quine = pl.read_csv("tests/data/quine.csv")

wage_panel_pd = wage_panel.load().set_index(["nr", "year"])


@pytest.fixture(scope="session")
def guerry_mod():
    return smf.ols("Literacy ~ Pop1831 * Desertion", guerry).fit()


@pytest.fixture(scope="session")
def impartiality_model():
    return smf.logit(
        "impartial ~ equal * democracy + continent", data=impartiality_df.to_pandas()
    ).fit()


@pytest.fixture(scope="session")
def penguins_model():
    mod = smf.ols(
        "body_mass_g ~ flipper_length_mm * species * bill_length_mm + island",
        data=penguins.to_pandas(),
    ).fit()
    return mod


@pytest.fixture(scope="session")
def penguins_mod_add():
    mod = smf.ols(
        "body_mass_g ~ flipper_length_mm * species * bill_length_mm * island",
        penguins.to_pandas(),
    ).fit()
    return mod


@pytest.fixture(scope="session")
def penguins_mod_5var():
    mod = smf.ols(
        "body_mass_g ~ flipper_length_mm * species * bill_length_mm * island * bill_depth_mm",
        penguins.to_pandas(),
    ).fit()
    return mod


@pytest.fixture(scope="session")
def mtcars_mod():
    mod = smf.ols("mpg ~ hp * wt * disp * cyl * qsec", data=mtcars).fit()
    return mod
