# Development

* Issue #90: Informative error on reserved keyword like 'group'.

# 0.0.8

* PyFixest: Fixed effects variables are recognized as categorical by `datagrid()`
* `MarginalEffectsDataFrame` class now has a `jacobian` attribute.

# 0.0.7

Breaking change:

* `datagridcf()` is deprecated. Use `datagrid(grid_type='counterfactual')` instead.

New:

* Support the `PyFixest` package. https://s3alfisc.github.io/pyfixest/
- `datagrid()` no longer requires specifying the `model` argument when called inside another marginaleffects function like `predictions()`.
* `eps_vcov` argument to control the step size in the computation of the Jacobian used for standard errors.
* plot_*() use `plotnine` instead of raw `matplotlib`
* plot_*() `condition` argument gets string shortcuts for numeric variables: "threenum", "fivenum", "minmax".
* `datagrid()` gets a `grid_type` argument: 'mean_or_mode', 'balance', 'counterfactual'
* Plot labels are sorted for clarity and consistency.
* `hypotheses()` function now supports raw models for conducting (non)-linear hypothesis tests on coefficients.

Misc:

* Refactor and several bug fixes in the `plot_*()` functions.
* Many bug fixes.
* Upgraded dependency on the `polars` package, with a shift from `.apply()` to `.map_*()` due to deprecation.
* Removed `pandas` dependency.


# 0.0.6

* `hypothesis` accepts a float or integer to specify a different null hypothesis.
* Better column order in printout when using `datagrid()` or `by`
* Version bump for dependencies.
* Equivalence test bug with duplicated column names.
* Minor bugs in plot_*() with unknown consequences.
* Linting.

# 0.0.5

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
