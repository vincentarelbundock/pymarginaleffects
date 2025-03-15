from .docs import DocsDetails, DocsParameters, docstring_returns
from .comparisons import comparisons


def slopes(
    model,
    variables=None,
    newdata=None,
    slope="dydx",
    vcov=True,
    conf_level=0.95,
    by=False,
    hypothesis=None,
    equivalence=None,
    wts=None,
    eps=1e-4,
    eps_vcov=None,
):
    if callable(newdata):
        newdata = newdata(model)

    assert isinstance(eps, float)

    if slope not in ["dydx", "eyex", "eydx", "dyex"]:
        raise ValueError("slope must be one of 'dydx', 'eyex', 'eydx', 'dyex'")

    out = comparisons(
        model=model,
        variables=variables,
        newdata=newdata,
        comparison=slope,
        vcov=vcov,
        conf_level=conf_level,
        by=by,
        hypothesis=hypothesis,
        equivalence=equivalence,
        wts=wts,
        eps=eps,
        eps_vcov=eps_vcov,
    )
    return out


def avg_slopes(
    model,
    variables=None,
    newdata=None,
    slope="dydx",
    vcov=True,
    conf_level=0.95,
    by=True,
    wts=None,
    hypothesis=None,
    equivalence=None,
    eps=1e-4,
    eps_vcov=None,
):
    if callable(newdata):
        newdata = newdata(model)

    if slope not in ["dydx", "eyex", "eydx", "dyex"]:
        raise ValueError("slope must be one of 'dydx', 'eyex', 'eydx', 'dyex'")
    out = slopes(
        model=model,
        variables=variables,
        newdata=newdata,
        slope=slope,
        vcov=vcov,
        conf_level=conf_level,
        by=by,
        wts=wts,
        hypothesis=hypothesis,
        equivalence=equivalence,
        eps=eps,
        eps_vcov=eps_vcov,
    )

    return out


docs_predictions = (
    """
# `slopes()`

`slopes()` and `avg_slopes()` estimate unit-level (conditional) partial derivative of the regression equation with respect to a regressor of interest.
    
* `slopes()`: unit-level (conditional) estimates.
* `avg_slopes()`: average (marginal) estimates.

The newdata argument and the `datagrid()` function can be used to control where statistics are evaluated in the predictor space: "at observed values", "at the mean", "at representative values", etc.

See the package website and vignette for examples:

- https://marginaleffects.com/chapters/slopes.html
- https://marginaleffects.com

## Parameters
"""
    + DocsParameters.docstring_model
    + """
`variables`: (str, list, dictionary) Specifies what variables (columns) to vary in order to make the slopes.

- str: Variable for which to compute the slopes for.
- NoneType: Slopes are computed for all regressors in the model object (can be slow)
"""
    + DocsParameters.docstring_newdata("slope")
    + DocsParameters.docstring_slope
    + DocsParameters.docstring_vcov
    + DocsParameters.docstring_conf_level
    + DocsParameters.docstring_by
    + DocsParameters.docstring_hypothesis
    + DocsParameters.docstring_equivalence
    + DocsParameters.docstring_wts
    + DocsParameters.docstring_eps
    + DocsParameters.docstring_eps_vcov
    + docstring_returns
    + """ 
## Details
"""
    + DocsDetails.docstring_tost
    + DocsDetails.docstring_order_of_operations
)


slopes.__doc__ = docs_predictions

avg_slopes.__doc__ = slopes.__doc__
