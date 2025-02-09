from docstring_inheritance import inherit_numpy_docstring
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
    """
    Estimate unit-level (conditional) partial derivative of the regression equation with respect to a regressor of interest.

    The newdata argument and the datagrid() function can be used to control where statistics are evaluated in the predictor space: "at observed values", "at the mean", "at representative values", etc.
    See the package website and vignette for examples:
        - https://marginaleffects.com/chapters/slopes.html
        - https://marginaleffects.com

    Parameters
    ----------
    variables : str, list, dictionary
        A string, list of strings, or dictionary of variables to compute slopes for. If `None`, slopes are computed for all regressors in the model object (can be slow). Acceptable values depend on the variable type. See the examples below.
        - List[str] or str: List of variable names to compute slopes for.
        - Dictionary: keys identify the subset of variables of interest, and values define locations between which the slope is computed. Acceptable values depend on the variable type:
            - Categorical variables:
                * "reference": Each factor level is compared to the factor reference (base) level
                * "all": All combinations of observed levels
                * "sequential": Each factor level is compared to the previous factor level
                * "pairwise": Each factor level is compared to all other levels
                * "minmax": The highest and lowest levels of a factor.
                * "revpairwise", "revreference", "revsequential": inverse of the corresponding hypotheses.
                * Vector of length 2 with the two values to compare.
            - Boolean variables:
                * `None`: contrast between True and False
            - Numeric variables:
                * Numeric of length 1: Contrast for a gap of `x`, computed at the observed value plus and minus `x / 2`. For example, estimating a `+1` contrast compares adjusted predictions when the regressor is equal to its observed value minus 0.5 and its observed value plus 0.5.
                * Numeric of length equal to the number of rows in `newdata`: Same as above, but the contrast can be customized for each row of `newdata`.
                * Numeric vector of length 2: Contrast between the 2nd element and the 1st element of the `x` vector.
                * Data frame with the same number of rows as `newdata`, with two columns of "low" and "high" values to compare.
                * Function which accepts a numeric vector and returns a data frame with two columns of "low" and "high" values to compare. See examples below.
                * "iqr": Contrast across the interquartile range of the regressor.
                * "sd": Contrast across one standard deviation around the regressor mean.
                * "2sd": Contrast across two standard deviations around the regressor mean.
                * "minmax": Contrast between the maximum and the minimum values of the regressor.
        - Examples:
            + `variables = {"gear" = "pairwise", "hp" = 10}`
            + `variables = {"gear" = "sequential", "hp" = [100, 120]}`
    newdata : polars or pandas DataFrame, or str
        Data frame or string specifying where statistics are evaluated in the predictor space. If `None`, unit-level slopes are computed for each observed value in the original dataset (empirical distribution).
        - Dataframe: should be created with datagrid() function
        - String:
            * "mean": Compute slopes at the mean of the regressor
            * "median": Compute slopes at the median of the regressor
            * "balanced": Slopes evaluated on a balanced grid with every combination of categories and numeric variables held at their means.
            * "tukey": Probably NotImplemented
            * "grid": Probably NotImplemented
    slope : str
        The type of slope or (semi-)elasticity to compute. Acceptable values are:
            - "dydx": dY/dX
            - "eyex": dY/dX * Y / X
            - "eydx": dY/dX * Y
            - "dyex": dY/dX / X

    """
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
    """
    Estimate average (marginal) partial derivative of the regression equation with respect to a regressor of interest.

    This function computes average partial derivatives across the sample or within groups. The newdata argument and
    the datagrid() function can be used to control where statistics are evaluated in the predictor space: "at observed values",
    "at the mean", "at representative values", etc.
    See the package website and vignette for examples:
        - https://marginaleffects.com/chapters/slopes.html
        - https://marginaleffects.com
    """
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


inherit_numpy_docstring(comparisons.__doc__, slopes)
inherit_numpy_docstring(slopes.__doc__, avg_slopes)
