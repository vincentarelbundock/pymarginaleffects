import polars as pl

from .docs import DocsDetails, DocsParameters

from .classes import MarginaleffectsDataFrame
from .equivalence import get_equivalence
from .hypothesis import get_hypothesis
from .sanity import sanitize_hypothesis_null, sanitize_vcov
from .sanitize_model import sanitize_model
from .uncertainty import get_jacobian, get_se, get_z_p_ci
from .utils import sort_columns
from .hypotheses_joint import joint_hypotheses


def hypotheses(
    model,
    hypothesis=None,
    conf_level=0.95,
    vcov=True,
    equivalence=None,
    eps_vcov=None,
    joint=False,
    joint_test="f",
):
    """
    # `hypotheses()`

    (Non-)Linear Tests for Null Hypotheses, Joint Hypotheses, Equivalence, Non Superiority, and Non Inferiority.

    This function calculates uncertainty estimates as first-order approximate standard errors for linear or non-linear
    functions of a vector of random variables with known or estimated covariance matrix. It emulates the behavior of
    the excellent and well-established `car::deltaMethod` and `car::linearHypothesis` functions in R, but it supports
    more models; requires fewer dependencies; expands the range of tests to equivalence and superiority/inferiority;
    and offers convenience features like robust standard errors.

    To learn more, visit the package website: <https://marginaleffects.com/>

    ## Parameters
    * model : object
        Model object estimated by `statsmodels`

    `hypothesis`: (str, int, float, numpy array) Specifies a hypothesis test or custom contrast

    * Number to specify the null hypothesis.
    * Numpy array with a number of rows equal to the number of estimates.
    * String equation with an equal sign and estimate number in b0, b1, b2, etc. format.
        - "b0 = b1"
        - "b0 - (b1 + b2) = 0"
    * Two-side formula like "ratio ~ reference"
        - Left-hand side: "ratio", "difference"
        - Right-hand side: 'reference', 'sequential', 'pairwise', 'revreference', 'revsequential', 'revpairwise'

    - int, float: The null hypothesis used in the computation of Z and p-values (before applying transform)
    - str:
        * equation specifying linear or non-linear hypothesis tests. Use the names of the model variables, or use `b0`, `b1` to identify the position of each parameter. The `b*` wildcard can be used to test hypotheses on all estimates. Examples:
            - `hp = drat`
            - `hp + drat = 12`
            - `b0 + b1 + b2 = 0`
            - `b* / b0 = 1`
        * one of the following hypothesis test strings:
            - `pairwise` and `revpairwise`: pairwise differences between estimates in each row.
            - `reference` and `revreference`: differences between the estimates in each row and the estimate in the first row.
            - `sequential` and `revsequential`: differences between an estimate and the estimate in the next row.
    - numpy.ndarray: Each column is a vector of weights. The output is the dot product between these vectors of weights and the vectors of estimates. e.g. `hypothesis=np.array([[1, 1, 2], [2, 2, 3]]).T`
    - See the Examples section and the vignette: https://marginaleffects.com/chapters/hypothesis.html

    `conf_level`: (float, default=0.95) Numeric value specifying the confidence level for the confidence intervals.

    `vcov`: (bool, np.ndarray, default=True) Type of uncertainty estimates to report (e.g. for robust standard errors). Acceptable values are:

    - `True`: Use the model's default covariance matrix.
    - `False`: Do not compute standard errors.
    - String: Literal indicating the kind of uncertainty estimates to return:
        - Heteroskedasticity-consistent: `"HC0"`, `"HC1"`, `"HC2"`, `"HC3"`.
    - np.ndarray: A custom square covariance matrix.

    `equivalence`: (list) List of 2 numeric values specifying the bounds used for the two-one-sided test (TOST) of equivalence, and for the non-inferiority and non-superiority tests. See the Details section below.

    `eps_vcov`: (float) optional custom value for the finite difference approximation of the jacobian matrix. By default, the function uses the square root of the machine epsilon.

    * joint: (bool, str, List[str], default = `False`) Specifies the joint test of statistical significance. The null hypothesis value can be set using the hypothesis argument.
        - `False`: Hypothesis are not tested jointly
        - `True`: Hypothesis are tested jointly
        - List[str]: Parameter names to be tested jointly as displayed by `mod.model.data.param_names`
        - List[int]: Parameter positions to test jointly where positions refer to the order specified by `mod.model.data.param_names`

    * joint_test: (str, default=`"f"`) Chooses the type of test between `"f"` and `"chisq"`


    ## Returns
    (MarginaleffectsDataFrame)
    * DataFrame containing the results of the hypothesis tests.

    ## Examples
    ```py
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
    ```
    ## Warnings
    * Warning #1: Tests are conducted directly on the scale defined by the `type` argument. For some models, it can make sense to conduct hypothesis or equivalence tests on the `"link"` scale instead of the `"response"` scale which is often the default.
    * Warning #2: For hypothesis tests on objects produced by the `marginaleffects` package, it is safer to use the `hypothesis` argument of the original function.
    * Warning #3: The tests assume that the `hypothesis` expression is (approximately) normally distributed, which for non-linear functions of the parameters may not be realistic. More reliable confidence intervals can be obtained using the `inferences()` (in R only) function with `method = "boot"`

    ## Details

    ### Two-One-Sided Test (TOST) of Equivalence

    The `equivalence` argument specifies the bounds used for the two-one-sided test (TOST) of equivalence, and for the non-inferiority and non-superiority tests. The first element specifies the lower bound, and the second element specifies the upper bound. If `None`, equivalence tests are not performed.
    """
    model = sanitize_model(model)

    if joint:
        out = joint_hypotheses(
            model, joint_index=joint, joint_test=joint_test, hypothesis=hypothesis
        )
        return out

    hypothesis_null = sanitize_hypothesis_null(hypothesis)
    V = sanitize_vcov(vcov, model)

    # estimands
    def fun(x):
        out = pl.DataFrame({"term": model.find_coef(), "estimate": x})
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
    else:
        J = None
    out = get_equivalence(out, equivalence=equivalence)
    out = sort_columns(out, by=None)
    out = MarginaleffectsDataFrame(out, conf_level=conf_level, jacobian=J)
    return out


hypotheses.__doc__ = (
    """
# `hypotheses()`

(Non-)Linear Tests for Null Hypotheses, Joint Hypotheses, Equivalence, Non Superiority, and Non Inferiority.

This function calculates uncertainty estimates as first-order approximate standard errors for linear or non-linear
functions of a vector of random variables with known or estimated covariance matrix. It emulates the behavior of
the excellent and well-established `car::deltaMethod` and `car::linearHypothesis` functions in R, but it supports
more models; requires fewer dependencies; expands the range of tests to equivalence and superiority/inferiority;
and offers convenience features like robust standard errors.

To learn more, visit the package website: <https://marginaleffects.com/>

## Parameters
* model : object
    Model object estimated by `statsmodels`
"""
    + DocsParameters.docstring_hypothesis
    + DocsParameters.docstring_conf_level
    + DocsParameters.docstring_vcov
    + DocsParameters.docstring_equivalence
    + DocsParameters.docstring_eps_vcov
    # add joint param docstring
    # add joint test param dosctring
    + """
* joint: (bool, str, List[str], default = `False`) Specifies the joint test of statistical significance. The null hypothesis value can be set using the hypothesis argument.
    - `False`: Hypothesis are not tested jointly
    - `True`: Hypothesis are tested jointly
    - List[str]: Parameter names to be tested jointly as displayed by `mod.model.data.param_names`
    - List[int]: Parameter positions to test jointly where positions refer to the order specified by `mod.model.data.param_names`
    
* joint_test: (str, default=`"f"`) Chooses the type of test between `"f"` and `"chisq"`


## Returns
(MarginaleffectsDataFrame)
* DataFrame containing the results of the hypothesis tests.

## Examples
```py
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
```
## Warnings
* Warning #1: Tests are conducted directly on the scale defined by the `type` argument. For some models, it can make sense to conduct hypothesis or equivalence tests on the `"link"` scale instead of the `"response"` scale which is often the default.
* Warning #2: For hypothesis tests on objects produced by the `marginaleffects` package, it is safer to use the `hypothesis` argument of the original function.
* Warning #3: The tests assume that the `hypothesis` expression is (approximately) normally distributed, which for non-linear functions of the parameters may not be realistic. More reliable confidence intervals can be obtained using the `inferences()` (in R only) function with `method = "boot"`

## Details
"""
    + DocsDetails.docstring_tost
)
