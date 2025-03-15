import polars as pl
import statsmodels.formula.api as smf
from polars.testing import assert_frame_equal

from marginaleffects import *
from tests.helpers import mtcars


mod = smf.ols("am ~ hp + wt + disp", data=mtcars).fit()

mod_without_intercept = smf.ols("am ~ 0 + hp + wt + disp", data=mtcars).fit()


def test_hypotheses_joint():
    hypo_py = hypotheses(mod, joint=["hp", "wt"])
    hypo_r = pl.read_csv("tests/r/test_hypotheses_joint_01.csv").rename(
        {"p.value": "p_value"}
    )
    assert_frame_equal(hypo_py, hypo_r)

    hypo_py = hypotheses(mod, joint=["hp", "disp"], joint_test="chisq")
    hypo_r = pl.read_csv("tests/r/test_hypotheses_joint_02.csv").rename(
        {"p.value": "p_value"}
    )
    assert_frame_equal(hypo_py, hypo_r)

    hypo_py = hypotheses(mod, joint=[1, 2])
    hypo_r = pl.read_csv("tests/r/test_hypotheses_joint_03.csv").rename(
        {"p.value": "p_value"}
    )
    assert_frame_equal(hypo_py, hypo_r)

    hypo_py = hypotheses(mod, joint=[0, 1, 2], hypothesis=[1, 2, 3])
    hypo_r = pl.read_csv("tests/r/test_hypotheses_joint_04.csv").rename(
        {"p.value": "p_value"}
    )
    hypo_r = hypo_r.cast({"p_value": pl.Float64})
    assert_frame_equal(hypo_py, hypo_r, check_exact=False, atol=0.0001)

    hypo_py = hypotheses(mod, joint=["Intercept", "disp", "wt"], hypothesis=4)
    hypo_r = pl.read_csv("tests/r/test_hypotheses_joint_05.csv").rename(
        {"p.value": "p_value"}
    )
    hypo_r = hypo_r.cast({"p_value": pl.Float64})
    assert_frame_equal(hypo_py, hypo_r, check_exact=False, atol=0.0001)

    hypo_py = hypotheses(mod_without_intercept, joint=[0, 1, 2])
    hypo_r = pl.read_csv("tests/r/test_hypotheses_joint_06.csv").rename(
        {"p.value": "p_value"}
    )
    assert_frame_equal(hypo_py, hypo_r)

    hypo_py = hypotheses(mod_without_intercept, joint=["hp", "wt"])
    hypo_r = pl.read_csv("tests/r/test_hypotheses_joint_07.csv").rename(
        {"p.value": "p_value"}
    )
    assert_frame_equal(hypo_py, hypo_r)
