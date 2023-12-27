import polars as pl

from .classes import MarginaleffectsDataFrame
from .equivalence import get_equivalence
from .hypothesis import get_hypothesis
from .sanity import sanitize_hypothesis_null, sanitize_vcov
from .sanitize_model import sanitize_model
from .uncertainty import get_jacobian, get_se, get_z_p_ci
from .utils import sort_columns


def hypotheses(
    model, hypothesis=None, conf_level=0.95, vcov=True, equivalence=None, eps_vcov=None
):
    """
    (Non-)Linear Tests for Null Hypotheses, Joint Hypotheses, Equivalence, Non Superiority, and Non Inferiority.

    This function calculates uncertainty estimates as first-order approximate standard errors for linear or non-linear
    functions of a vector of random variables with known or estimated covariance matrix. It emulates the behavior of
    the excellent and well-established `car::deltaMethod` and `car::linearHypothesis` functions in R, but it supports
    more models; requires fewer dependencies; expands the range of tests to equivalence and superiority/inferiority;
    and offers convenience features like robust standard errors.

    To learn more, visit the package website: <https://marginaleffects.com/>

    Warning #1: Tests are conducted directly on the scale defined by the `type` argument. For some models, it can make
    sense to conduct hypothesis or equivalence tests on the `"link"` scale instead of the `"response"` scale which is
    often the default.

    Warning #2: For hypothesis tests on objects produced by the `marginaleffects` package, it is safer to use the
    `hypothesis` argument of the original function. Using `hypotheses()` may not work in certain environments, in lists,
    or when working programmatically with *apply style functions.

    Warning #3: The tests assume that the `hypothesis` expression is (approximately) normally distributed, which for
    non-linear functions of the parameters may not be realistic. More reliable confidence intervals can be obtained using
    the `inferences()` function with `method = "boot"`.

    Parameters:
    model : object
        Model object estimated by `statsmodels`
    hypothesis : str formula, int, float, or optional
        The null hypothesis value. Default is None.
    conf_level : float, optional
        Confidence level for the hypothesis test. Default is 0.95.
    vcov : bool, optional
        Whether to use the covariance matrix in the hypothesis test. Default is True.
    equivalence : tuple, optional
        The equivalence range for the hypothesis test. Default is None.

    Returns:
    MarginaleffectsDataFrame
        A DataFrame containing the results of the hypothesis tests.

    Examples:

        # When `hypothesis` is `None`, `hypotheses()` returns a DataFrame of parameters
        hypotheses(model)

        # A different null hypothesis
        hypotheses(model, hypothesis = 3)

        # Test of equality between coefficients
        hypotheses(model, hypothesis="param1 = param2")

        # Non-linear function
        hypotheses(model, hypothesis="exp(param1 + param2) = 0.1")

        # Robust standard errors
        hypotheses(model, hypothesis="param1 = param2", vcov="HC3")

        # Equivalence, non-inferiority, and non-superiority tests
        hypotheses(model, equivalence=(0, 10))
    """

    model = sanitize_model(model)
    V = sanitize_vcov(vcov, model)

    hypothesis_null = sanitize_hypothesis_null(hypothesis)

    # estimands
    def fun(x):
        out = pl.DataFrame({"term": model.get_coef_names(), "estimate": x})
        out = get_hypothesis(out, hypothesis=hypothesis)
        return out

    out = fun(model.coef)
    if vcov is not None:
        J = get_jacobian(fun, model.coef, eps_vcov=eps_vcov)
        se = get_se(J, V)
        out = out.with_columns(pl.Series(se).alias("std_error"))
        out = get_z_p_ci(
            out, model, conf_level=conf_level, hypothesis_null=hypothesis_null
        )
    out = get_equivalence(out, equivalence=equivalence)
    out = sort_columns(out, by=None)
    out = MarginaleffectsDataFrame(out, conf_level=conf_level)
    return out
