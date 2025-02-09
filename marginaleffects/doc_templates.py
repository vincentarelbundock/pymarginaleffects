def _template_returns():
    """
    Returns
    -------
    DataFrame
        The functions return a data.frame with the following columns:
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
