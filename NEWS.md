# Dev

* `MarginaleffectsDataFrame` class inherits from `pl.DataFrame` for better printing and to host useful attributes.

# 0.0.2

Misc:

* Better step size selection for the numerical derivatives used to compute delta method standard errors.

Bugs:

* When newdata was an unseen dataframe, out.columns would be referenced in sanity.py prior to assignment. Thanks to @Vinnie-Palazeti for PR #25.


# 0.0.1

Initial release