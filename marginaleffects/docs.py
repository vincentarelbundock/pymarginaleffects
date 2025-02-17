import os
import inspect
import marginaleffects


class DocsParameters:
    docstring_hypothesis = """
* hypothesis : (str, numpy array)
    String specifying a numeric value specifying the null hypothesis used for computing p-values.
"""
    docstring_by = """
* by : (bool, List[str], optional)
    A logical value or a list of column names in `newdata`. If `True`, estimate is aggregated across the whole dataset. If a list is provided, estimates are aggregated for each unique combination of values in the columns.
"""
    docstring_conf_level = """
* conf_level : (float)
    Numeric value specifying the confidence level for the confidence intervals. Default is 0.95.
"""
    docstring_wts = """
* wts : (str, optional)
    Column name of weights to use for marginalization. Must be a column in `newdata`.
"""
    docstring_vcov = """
* vcov : (bool, np.ndarray, optional)
    Type of uncertainty estimates to report (e.g. for robust standard errors). Acceptable values are:
    - `True`: Use the model's default covariance matrix.
    - `False`: Do not compute standard errors.
    - np.ndarray: A custom square covariance matrix.
"""
    docstring_equivalence = """
* equivalence : (list)
    - List of 2 numeric values specifying the bounds used for the two-one-sided test (TOST) of equivalence, and for the non-inferiority and non-superiority tests. See the Details section below.
"""
    docstring_transform = """
* transform : (function)
    Function specifying a transformation applied to unit-level estimates and confidence intervals just before the function returns results. Functions must accept a full column (series) of a Polars data frame and return a corresponding series of the same length. Ex:
        - `transform = numpy.exp`
        - `transform = lambda x: x.exp()`
        - `transform = lambda x: x.map_elements()`
"""

    docstring_newdata = """
* newdata : (polars or pandas DataFrame, or str)
Data frame or string specifying where statistics are evaluated in the predictor space. If `None`, unit-level contrasts are computed for each observed value in the original dataset (empirical distribution).
    - Dataframe: should be created with datagrid() function
    - String:
        * "mean": Compute comparisons at the mean of the regressor
        * "median": Compute comparisons at the median of the regressor
        * "balanced": Comparisons evaluated on a balanced grid with every combination of categories and numeric variables held at their means.
        * "tukey": Probably NotImplemented
        * "grid": Probably NotImplemented
"""
    docstring_model = """
* model : (object Model). 
Object fitted using the `statsmodels` formula API.
"""
    docstring_eps_vcov = """
* eps_vcov : float, optional
    custom value for the finite difference approximation of the jacobian matrix. by default, the function uses the square root of the machine epsilon.
"""
    docstring_variables = """
* variables : (str, list, dictionary)
Specifies what variables (columns) to vary in order to make the comparison.
If `None`, comparisons are computed for all regressors in the model object (can be slow). Acceptable values depend on the variable type. See the examples below.
    - List[str] or str: List of variable names to compute comparisons for.
    - Dictionary: keys identify the subset of variables of interest, and values define the type of contrast to compute. Acceptable values depend on the variable type:
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
"""


docstring_returns = """

## Returns

A Polars DataFrame with the following columns:

- term: the name of the variable.
- contrast: the comparison method used.
- estimate: the estimated contrast, difference, ratio, or other transformation between pairs of predictions.
- std_error: the standard error of the estimate.
- statistic: the test statistic (estimate / std.error).
- p_value: the p-value of the test.
- s_value: Shannon transform of the p value.
- conf_low: the lower confidence interval bound.
- conf_high: the upper confidence interval bound.

"""


class DocsDetails:
    docstring_tost = """
- The `equivalence` argument specifies the bounds used for the two-one-sided test (TOST) of equivalence, and for the non-inferiority and non-superiority tests. The first element specifies the lower bound, and the second element specifies the upper bound. If `None`, equivalence tests are not performed.
"""

    docstring_order_of_operations = """
- Order of operations. Behind the scenes, the arguments of `marginaleffects` functions are evaluated in this order:
    1. `newdata`
    2. `variables`
    3. `comparison` and `slope`
    4. `by`
    5. `vcov`
    6. `hypothesis`
    7. `transform`
"""


def docstrings_to_qmd(output_dir: str):
    """
    Loops over every name in marginaleffects.__all__ and writes the
    function's docstring (if it is indeed a function) to a .qmd file
    in the specified directory.

    Parameters
    ----------
    output_dir : str
        The directory to which the .qmd files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    for name in getattr(marginaleffects, "__all__", []):
        # Retrieve the object by name
        obj = getattr(marginaleffects, name, None)

        # Check if the object is a function
        if obj is not None and inspect.isfunction(obj):
            docstring = inspect.getdoc(obj) or ""

            # Construct the filepath as "output_dir/name.qmd"
            filepath = os.path.join(output_dir, f"{name}.qmd")

            # Write the docstring to the file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(docstring)


docstrings_to_qmd("qmd_files")
