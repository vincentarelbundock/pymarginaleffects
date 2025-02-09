import os
import inspect
import marginaleffects


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