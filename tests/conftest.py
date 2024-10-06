import pytest
import statsmodels.formula.api as smf
import polars as pl

mtcars_df = pl.read_csv(
    "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv"
)

penguins = pl.read_csv(
    "https://vincentarelbundock.github.io/Rdatasets/csv/palmerpenguins/penguins.csv",
    null_values="NA",
).drop_nulls()


@pytest.fixture(scope="session")
def mtcars():
    mtcars = pl.read_csv(
        "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv"
    )


@pytest.fixture(scope="session")
def model():
    mod = smf.ols(
        "body_mass_g ~ flipper_length_mm * species * bill_length_mm + island",
        data=penguins.to_pandas(),
    ).fit()
    return mod


@pytest.fixture(scope="session")
def mod():
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
    mod = smf.ols("mpg ~ hp * wt * disp * cyl * qsec", data=mtcars_df).fit()
    return mod
