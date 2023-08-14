# dev

* `predictions()` supports categorical predictors when `newdata` does not include all levels (internal padding).
* Better sorting of output, using the `by` argument.

# 0.0.4

* New function: `datagridcf()`
* `predictions()` supports categorical predictors when `newdata` does not include all levels (internal padding).

# 0.0.3

Breaking change:

* Rename argument to match `R` `marginaleffects`: `conf_int` -> `conf_level`

Misc:

* `MarginaleffectsDataFrame` class inherits from `pl.DataFrame` for better printing and to host useful attributes.

# 0.0.2

Misc:

* Better step size selection for the numerical derivatives used to compute delta method standard errors.

Bugs:

* When newdata was an unseen dataframe, out.columns would be referenced in sanity.py prior to assignment. Thanks to @Vinnie-Palazeti for PR #25.


# 0.0.1

Initial release