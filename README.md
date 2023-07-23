# `marginaleffects` for Python

The `marginaleffects` package allows `Python` users to compute and plot
three principal quantities of interest: (1) predictions, (2)
comparisons, and (3) slopes. In addition, the package includes a
convenience function to compute a fourth estimand, “marginal means”,
which is a special case of averaged predictions. `marginaleffects` can
also average (or “marginalize”) unit-level (or “conditional”) estimates
of all those quantities, and conduct hypothesis tests on them.

## WARNING

This is an *alpha* version of the package, released to gather feedback,
feature requests, and bug reports from potential users. This version
includes known bugs. There are also known discrepancies between the
numerical results produced in Python and R. Please report any issues you
encounter here:
https://github.com/vincentarelbundock/pymarginaleffects/issues

## Supported models

There is a good chance that this package will work with (nearly) all the
models supported by [the `statsmodels` formula
API,](https://www.statsmodels.org/stable/api.html#statsmodels-formula-api)
ex: `ols`, `probit`, `logit`, `mnlogit`, `quantreg`, `poisson`,
`negativebinomial`, `mixedlm`, `rlm`, etc. However, the package has only
been tested with a subset of those, and some weirdness remains. Again:
this is *alpha* software; it should not be used in critical applications
yet.

## Installation

Install the latest PyPi release:

``` python
pip install marginaleffects
```

## Estimands: Predictions, Comparisons, and Slopes

## Definitions

[*Predictions*:](predictions.html)

> The outcome predicted by a fitted model on a specified scale for a
> given combination of values of the predictor variables, such as their
> observed values, their means, or factor levels. a.k.a. Fitted values,
> adjusted predictions. `predictions()`, `avg_predictions()`,
> `plot_predictions()`.

[*Comparisons*:](comparisons.html)

> Compare the predictions made by a model for different regressor values
> (e.g., college graduates vs. others): contrasts, differences, risk
> ratios, odds, etc. `comparisons()`, `avg_comparisons()`,
> `plot_comparisons()`.

[*Slopes*:](slopes.html)

> Partial derivative of the regression equation with respect to a
> regressor of interest. a.k.a. Marginal effects, trends. `slopes()`,
> `avg_slopes()`, `plot_slopes()`.

[Hypothesis and Equivalence Tests:](hypothesis.html)

> Hypothesis and equivalence tests can be conducted on linear or
> non-linear functions of model coefficients, or on any of the
> quantities computed by the `marginaleffects` packages (predictions,
> slopes, comparisons, marginal means, etc.). Uncertainy estimates can
> be obtained via the delta method (with or without robust standard
> errors), bootstrap, or simulation.

Predictions, comparisons, and slopes are fundamentally unit-level (or
“conditional”) quantities. Except in the simplest linear case, estimates
will typically vary based on the values of all the regressors in a
model. Each of the observations in a dataset is thus associated with its
own prediction, comparison, and slope estimates. Below, we will see that
it can be useful to marginalize (or “average over”) unit-level estimates
to report an “average prediction”, “average comparison”, or “average
slope”.

We now apply `marginaleffects` functions to compute each of the
estimands described above. First, we fit a linear regression model with
multiplicative interactions:

#### Predictions

``` python
import numpy as np
import polars as pl
from marginaleffects import *
import statsmodels.formula.api as smf
mtcars = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv")
mod = smf.ols("mpg ~ hp * wt * am", data = mtcars).fit()

print(mod.summary().as_text())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                    mpg   R-squared:                       0.896
    Model:                            OLS   Adj. R-squared:                  0.866
    Method:                 Least Squares   F-statistic:                     29.55
    Date:                Sun, 23 Jul 2023   Prob (F-statistic):           2.60e-10
    Time:                        11:32:26   Log-Likelihood:                -66.158
    No. Observations:                  32   AIC:                             148.3
    Df Residuals:                      24   BIC:                             160.0
    Df Model:                           7                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     40.3272     13.008      3.100      0.005      13.480      67.175
    hp            -0.0888      0.065     -1.372      0.183      -0.222       0.045
    wt            -4.7968      4.002     -1.199      0.242     -13.056       3.462
    hp:wt          0.0145      0.019      0.755      0.458      -0.025       0.054
    am            12.8371     14.222      0.903      0.376     -16.517      42.191
    hp:am         -0.0326      0.089     -0.366      0.717      -0.216       0.151
    wt:am         -5.3620      4.597     -1.166      0.255     -14.851       4.127
    hp:wt:am       0.0178      0.026      0.680      0.503      -0.036       0.072
    ==============================================================================
    Omnibus:                        1.875   Durbin-Watson:                   2.205
    Prob(Omnibus):                  0.392   Jarque-Bera (JB):                1.588
    Skew:                           0.528   Prob(JB):                        0.452
    Kurtosis:                       2.721   Cond. No.                     3.32e+04
    ==============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 3.32e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.

Then, we call the `predictions()` function. As noted above, predictions
are unit-level estimates, so there is one specific prediction per
observation. By default, the `predictions()` function makes one
prediction per observation in the dataset that was used to fit the
original model. Since `mtcars` has 32 rows, the `predictions()` outcome
also has 32 rows:

``` python
pre = predictions(mod)

pre.shape

print(pre.head())
```

    | Estimate  | Std.Error | z         | P(>|z|)    | S         | [         | ]         |
    |-----------|-----------|-----------|------------|-----------|-----------|-----------|
    | 22.488569 | 0.884149  | 25.435278 | 0.0        | inf       | 20.663775 | 24.313362 |
    | 20.801859 | 1.194205  | 17.419002 | 3.9968e-15 | 47.830075 | 18.337141 | 23.266577 |
    | 25.264652 | 0.708531  | 35.657806 | 0.0        | inf       | 23.802316 | 26.726987 |
    | 20.255492 | 0.704464  | 28.753051 | 0.0        | inf       | 18.80155  | 21.709435 |
    | 16.997817 | 0.711866  | 23.87784  | 0.0        | inf       | 15.528599 | 18.467036 |
    Columns: rowid, estimate, std_error, statistic, p_value, s_value, conf_low, conf_high, , mpg, cyl, disp, hp, drat, wt, qsec, vs, am, gear, carb

#### Comparisons: Differences, Ratios, Log-Odds, Lift, etc.

Now, we use the `comparisons()` function to compute the difference in
predicted outcome when each of the predictors is incremented by 1 unit
(one predictor at a time, holding all others constant). Once again,
comparisons are unit-level quantities. And since there are 3 predictors
in the model and our data has 32 rows, we obtain 96 comparisons:

``` python
cmp = comparisons(mod)

cmp.shape

print(cmp.head())
```

    | Term | Contrast | Estimate  | Std.Error | … | P(>|z|)  | S        | [         | ]        |
    |------|----------|-----------|-----------|---|----------|----------|-----------|----------|
    | am   | 1 - 0    | 0.325174  | 1.682201  | … | 0.848349 | 0.237271 | -3.146718 | 3.797066 |
    | am   | 1 - 0    | -0.543864 | 1.568211  | … | 0.73176  | 0.450558 | -3.780491 | 2.692764 |
    | am   | 1 - 0    | 1.200713  | 2.347556  | … | 0.613693 | 0.70441  | -3.644405 | 6.045831 |
    | am   | 1 - 0    | -1.70258  | 1.86713   | … | 0.370906 | 1.430875 | -5.556147 | 2.150986 |
    | am   | 1 - 0    | -0.614695 | 1.680809  | … | 0.717782 | 0.478381 | -4.083713 | 2.854324 |
    Columns: rowid, term, contrast, estimate, std_error, statistic, p_value, s_value, conf_low, conf_high, predicted, predicted_lo, predicted_hi, , mpg, cyl, disp, hp, drat, wt, qsec, vs, am, gear, carb

The `comparisons()` function allows customized queries. For example,
what happens to the predicted outcome when the `hp` variable increases
from 100 to 120?

``` python
cmp = comparisons(mod, variables = {"hp": [120, 100]})
print(cmp)
```

    | Term | Contrast  | Estimate | Std.Error | … | P(>|z|)  | S        | 2.5%      | 97.5%    |
    |------|-----------|----------|-----------|---|----------|----------|-----------|----------|
    | hp   | 100 - 120 | 0.738111 | 0.370034  | … | 0.057551 | 4.119022 | -0.025602 | 1.501824 |
    | hp   | 100 - 120 | 0.573787 | 0.312572  | … | 0.078824 | 3.665214 | -0.07133  | 1.218905 |
    | hp   | 100 - 120 | 0.931433 | 0.451743  | … | 0.050209 | 4.315922 | -0.000918 | 1.863785 |
    | hp   | 100 - 120 | 0.845426 | 0.265656  | … | 0.004008 | 7.963035 | 0.297139  | 1.393712 |
    | …    | …         | …        | …         | … | …        | …        | …         | …        |
    | hp   | 100 - 120 | 0.383687 | 0.269795  | … | 0.167851 | 2.574744 | -0.173142 | 0.940516 |
    | hp   | 100 - 120 | 0.64145  | 0.334463  | … | 0.067105 | 3.897433 | -0.048849 | 1.331749 |
    | hp   | 100 - 120 | 0.125924 | 0.272165  | … | 0.647765 | 0.626459 | -0.435797 | 0.687645 |
    | hp   | 100 - 120 | 0.635006 | 0.332261  | … | 0.067998 | 3.878372 | -0.050746 | 1.320758 |
    Columns: rowid, term, contrast, estimate, std_error, statistic, p_value, s_value, conf_low, conf_high, predicted, predicted_lo, predicted_hi, , mpg, cyl, disp, hp, drat, wt, qsec, vs, am, gear, carb

What happens to the predicted outcome when the `wt` variable increases
by 1 standard deviation about its mean?

``` python
cmp = comparisons(mod, variables = {"hp": "sd"})
print(cmp)
```

    | Term | Contrast        | Estimate  | Std.Error | … | P(>|z|)  | S        | 2.5%      | 97.5%     |
    |------|-----------------|-----------|-----------|---|----------|----------|-----------|-----------|
    | hp   | +68.56286848932 | -2.530351 | 1.268531  | … | 0.057551 | 4.119015 | -5.14847  | 0.087769  |
    |      | 059             |           |           |   |          |          |           |           |
    | hp   | +68.56286848932 | -1.967025 | 1.071543  | … | 0.078825 | 3.665211 | -4.178581 | 0.244531  |
    |      | 059             |           |           |   |          |          |           |           |
    | hp   | +68.56286848932 | -3.193087 | 1.54864   | … | 0.050209 | 4.315921 | -6.389322 | 0.003148  |
    |      | 059             |           |           |   |          |          |           |           |
    | hp   | +68.56286848932 | -2.89824  | 0.910706  | … | 0.004008 | 7.963036 | -4.777845 | -1.018636 |
    |      | 059             |           |           |   |          |          |           |           |
    | …    | …               | …         | …         | … | …        | …        | …         | …         |
    | hp   | +68.56286848932 | -1.315334 | 0.924895  | … | 0.167851 | 2.574743 | -3.224224 | 0.593556  |
    |      | 059             |           |           |   |          |          |           |           |
    | hp   | +68.56286848932 | -2.198983 | 1.146589  | … | 0.067105 | 3.897431 | -4.565426 | 0.167461  |
    |      | 059             |           |           |   |          |          |           |           |
    | hp   | +68.56286848932 | -0.431686 | 0.933021  | … | 0.647764 | 0.626459 | -2.357346 | 1.493974  |
    |      | 059             |           |           |   |          |          |           |           |
    | hp   | +68.56286848932 | -2.176891 | 1.139037  | … | 0.067998 | 3.87837  | -4.527749 | 0.173966  |
    |      | 059             |           |           |   |          |          |           |           |
    Columns: rowid, term, contrast, estimate, std_error, statistic, p_value, s_value, conf_low, conf_high, predicted, predicted_lo, predicted_hi, , mpg, cyl, disp, hp, drat, wt, qsec, vs, am, gear, carb

The `comparisons()` function also allows users to specify arbitrary
functions of predictions, with the `comparison` argument. For example,
what is the average ratio between predicted Miles per Gallon after an
increase of 50 units in Horsepower?

``` python
cmp = comparisons(
  mod,
  variables = {"hp": 50},
  comparison = "ratioavg")
print(cmp)
```

    | Term | Contrast | Estimate | Std.Error | … | P(>|z|) | S   | 2.5%     | 97.5%   |
    |------|----------|----------|-----------|---|---------|-----|----------|---------|
    | hp   | +50      | 0.909534 | 0.02906   | … | 0.0     | inf | 0.849557 | 0.96951 |
    Columns: term, contrast, estimate, std_error, statistic, p_value, s_value, conf_low, conf_high

#### Slopes: Derivatives and elasticities

Consider a logistic regression model with a single predictor:

``` python
url = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv"
mtcars = pl.read_csv(url)
mod = smf.logit("am ~ mpg", data = mtcars).fit()
```

    Optimization terminated successfully.
             Current function value: 0.463674
             Iterations 6

We can estimate the slope of the prediction function with respect to the
`mpg` variable at any point in the data space. For example, what is the
slope of the prediction function at `mpg = 24`?

``` python
mfx = slopes(mod, newdata = datagrid(mpg = 24, newdata = mtcars))
print(mfx)
```

This is equivalent to the result we obtain by taking the analytical
derivative using the chain rule:

``` python
from scipy.stats import logistic
beta_0 = mod.params.iloc[0]
beta_1 = mod.params.iloc[1]
print(beta_1 * logistic.pdf(beta_0 + beta_1 * 24))
```

    0.06653436463892946

This computes a “marginal effect (or slope) at the mean” or “at the
median”, that is, when all covariates are held at their mean or median
values:

``` python
mfx = slopes(mod, newdata = "mean")
print(mfx)
```

    | Term | Contrast | Estimate | Std.Error | … | P(>|z|)  | S        | 2.5%     | 97.5%   |
    |------|----------|----------|-----------|---|----------|----------|----------|---------|
    | mpg  | dY/dX    | 0.073235 | 0.028324  | … | 0.014821 | 6.076202 | 0.015391 | 0.13108 |
    Columns: term, contrast, estimate, std_error, statistic, p_value, s_value, conf_low, conf_high

``` python
mfx = slopes(mod, newdata = "median")
print(mfx)
```

    | Term | Contrast | Estimate | Std.Error | … | P(>|z|)  | S        | 2.5%     | 97.5%   |
    |------|----------|----------|-----------|---|----------|----------|----------|---------|
    | mpg  | dY/dX    | 0.067875 | 0.025308  | … | 0.011785 | 6.406929 | 0.016189 | 0.11956 |
    Columns: term, contrast, estimate, std_error, statistic, p_value, s_value, conf_low, conf_high

We can also compute an “average slope” or “average marginaleffects”

``` python
mfx = avg_slopes(mod)
print(mfx)
```

    | Term | Contrast    | Estimate | Std.Error | … | P(>|z|)  | S         | 2.5%     | 97.5%    |
    |------|-------------|----------|-----------|---|----------|-----------|----------|----------|
    | mpg  | mean(dY/dX) | 0.046486 | 0.008862  | … | 0.000012 | 16.391095 | 0.028388 | 0.064584 |
    Columns: term, contrast, estimate, std_error, statistic, p_value, s_value, conf_low, conf_high

Which again is equivalent to the analytical result:

``` python
np.mean(beta_1 * logistic.pdf(beta_0 + beta_1 * mtcars["mpg"]))
```

    0.04648596405936302

## Grid

Predictions, comparisons, and slopes are typically “conditional”
quantities which depend on the values of all the predictors in the
model. By default, `marginaleffects` functions estimate quantities of
interest for the empirical distribution of the data (i.e., for each row
of the original dataset). However, users can specify the exact values of
the predictors they want to investigate by using the `newdata` argument.

`newdata` accepts data frames like this:

``` python
pre = predictions(mod, newdata = mtcars.tail(2))
print(pre)
```

    | Estimate | Std.Error | z        | P(>|z|)  | S        | 2.5%      | 97.5%    |
    |----------|-----------|----------|----------|----------|-----------|----------|
    | 0.119402 | 0.07784   | 1.533947 | 0.135522 | 2.883398 | -0.039568 | 0.278372 |
    | 0.49172  | 0.119613  | 4.110934 | 0.000281 | 11.79581 | 0.247438  | 0.736002 |
    Columns: rowid, estimate, std_error, statistic, p_value, s_value, conf_low, conf_high, , mpg, cyl, disp, hp, drat, wt, qsec, vs, am, gear, carb

The [`datagrid` function gives us a powerful way to define a grid of
predictors.](https://vincentarelbundock.github.io/marginaleffects/reference/datagrid.html)
All the variables not mentioned explicitly in `datagrid()` are fixed to
their mean or mode:

``` python
pre = predictions(
  mod,
  newdata = datagrid(
    newdata = mtcars,
    am = [0, 1],
    wt = [mtcars["wt"].max(), mtcars["wt"].min()]))

print(pre)
```

    | Estimate | Std.Error | z        | P(>|z|)  | S        | 2.5%     | 97.5%    |
    |----------|-----------|----------|----------|----------|----------|----------|
    | 0.3929   | 0.108367  | 3.625655 | 0.001056 | 9.887147 | 0.171586 | 0.614214 |
    | 0.3929   | 0.108367  | 3.625655 | 0.001056 | 9.887147 | 0.171586 | 0.614214 |
    | 0.3929   | 0.108367  | 3.625655 | 0.001056 | 9.887147 | 0.171586 | 0.614214 |
    | 0.3929   | 0.108367  | 3.625655 | 0.001056 | 9.887147 | 0.171586 | 0.614214 |
    Columns: rowid, estimate, std_error, statistic, p_value, s_value, conf_low, conf_high, am, wt, , mpg, cyl, disp, hp, drat, qsec, vs, gear, carb

## Averaging

Since predictions, comparisons, and slopes are conditional quantities,
they can be a bit unwieldy. Often, it can be useful to report a
one-number summary instead of one estimate per observation. Instead of
presenting “conditional” estimates, some methodologists recommend
reporting “marginal” estimates, that is, an average of unit-level
estimates.

(This use of the word “marginal” as “averaging” should not be confused
with the term “marginal effect” which, in the econometrics tradition,
corresponds to a partial derivative, or the effect of a “small/marginal”
change.)

To marginalize (average over) our unit-level estimates, we can use the
`by` argument or the one of the convenience functions:
`avg_predictions()`, `avg_comparisons()`, or `avg_slopes()`. For
example, both of these commands give us the same result: the average
predicted outcome in the `mtcars` dataset:

``` python
pre = avg_predictions(mod)
print(pre)
```

    | Estimate | Std.Error | z        | P(>|z|)  | S         | 2.5%     | 97.5%    |
    |----------|-----------|----------|----------|-----------|----------|----------|
    | 0.40625  | 0.068785  | 5.906042 | 0.000002 | 19.072753 | 0.265771 | 0.546729 |
    Columns: estimate, std_error, statistic, p_value, s_value, conf_low, conf_high

This is equivalent to manual computation by:

``` python
np.mean(mod.predict())
```

    0.40624999999999994

The main `marginaleffects` functions all include a `by` argument, which
allows us to marginalize within sub-groups of the data. For example,

``` python
cmp = avg_comparisons(mod, by = "am")
print(cmp)
```

    | am | Term | Contrast | Estimate | … | P(>|z|)   | S         | 2.5%     | 97.5%    |
    |----|------|----------|----------|---|-----------|-----------|----------|----------|
    | 1  | mpg  | +1       | 0.044926 | … | 1.4403e-7 | 22.727144 | 0.031476 | 0.058376 |
    | 0  | mpg  | +1       | 0.04751  | … | 0.000285  | 11.77591  | 0.023879 | 0.071141 |
    Columns: am, term, contrast, estimate, std_error, statistic, p_value, s_value, conf_low, conf_high

Marginal Means are a special case of predictions, which are marginalized
(or averaged) across a balanced grid of categorical predictors. To
illustrate, we estimate a new model with categorical predictors:

``` python
dat = mtcars \
  .with_columns(
    pl.col("am").cast(pl.Boolean),
    pl.col("cyl").cast(pl.Utf8)
  )
mod_cat = smf.ols("mpg ~ am + cyl + hp", data = dat).fit()
```

We can compute marginal means manually using the functions already
described:

``` python
pre = avg_predictions(
  mod_cat,
  newdata = datagrid(
    newdata = dat,
    cyl = dat["cyl"].unique(),
    am = dat["am"].unique()),
  by = "am")

print(pre)
```

``` python
cmp = avg_comparisons(mod_cat)
print(cmp)
```

    | Term | Contrast     | Estimate  | Std.Error | … | P(>|z|)  | S        | 2.5%      | 97.5%     |
    |------|--------------|-----------|-----------|---|----------|----------|-----------|-----------|
    | hp   | +1           | -0.044244 | 0.014576  | … | 0.005266 | 7.56902  | -0.074151 | -0.014337 |
    | am   | mean(True) - | 4.157856  | 1.25655   | … | 0.00266  | 8.55446  | 1.579628  | 6.736085  |
    |      | mean(False)  |           |           |   |          |          |           |           |
    | cyl  | 6 - 4        | -3.924578 | 1.537515  | … | 0.016663 | 5.907182 | -7.079299 | -0.769858 |
    | cyl  | 8 - 4        | -3.533414 | 2.502788  | … | 0.169433 | 2.561213 | -8.668711 | 1.601883  |
    Columns: term, contrast, estimate, std_error, statistic, p_value, s_value, conf_low, conf_high

## Hypothesis and equivalence tests

The `hypotheses()` function and the `hypothesis` argument can be used to
conduct linear and non-linear hypothesis tests on model coefficients, or
on any of the quantities computed by the functions introduced above.

Consider this model:

``` python
mod = smf.ols("mpg ~ qsec * drat", data = mtcars).fit()
mod.params
```

    Intercept    12.337199
    qsec         -1.024118
    drat         -3.437146
    qsec:drat     0.597315
    dtype: float64

Can we reject the null hypothesis that the `drat` coefficient is 2 times
the size of the `qsec` coefficient?

``` python
hyp = hypotheses(mod, "b3 = 2 * b2")
print(hyp)
```

    | term    | estimate  | std_error | statistic | p_value  | s_value  | conf_low   | conf_high |
    |---------|-----------|-----------|-----------|----------|----------|------------|-----------|
    | b3=2*b2 | -1.388909 | 10.77593  | -0.12889  | 0.898366 | 0.154625 | -23.462402 | 20.684583 |

The main functions in `marginaleffects` all have a `hypothesis`
argument, which means that we can do complex model testing. For example,
consider two slope estimates:

``` python
range = lambda x: [x.max(), x.min()]
cmp = comparisons(
  mod,
  variables = "drat",
  newdata = datagrid(newdata = mtcars, qsec = range(mtcars["qsec"])))
print(cmp)
```

    | Term | Contrast | Estimate  | Std.Error | … | P(>|z|)  | S        | 2.5%      | 97.5%     |
    |------|----------|-----------|-----------|---|----------|----------|-----------|-----------|
    | drat | +1       | 10.241374 | 5.161432  | … | 0.057112 | 4.130058 | -0.33134  | 20.814088 |
    | drat | +1       | 5.223926  | 3.791069  | … | 0.17913  | 2.480917 | -2.541727 | 12.989578 |
    Columns: rowid, term, contrast, estimate, std_error, statistic, p_value, s_value, conf_low, conf_high, predicted, predicted_lo, predicted_hi, qsec, , mpg, cyl, disp, hp, drat, wt, vs, am, gear, carb

Are these two contrasts significantly different from one another? To
test this, we can use the `hypothesis` argument:

``` python
cmp = comparisons(
  mod,
  hypothesis = "b1 = b2",
  variables = "drat",
  newdata = datagrid(newdata = mtcars, qsec = range(mtcars["qsec"])))
print(cmp)
```

    | Term  | Estimate | Std.Error | z        | P(>|z|)  | S        | 2.5%       | 97.5%     |
    |-------|----------|-----------|----------|----------|----------|------------|-----------|
    | b1=b2 | 5.017448 | 8.519298  | 0.588951 | 0.560616 | 0.834915 | -12.433542 | 22.468439 |
    Columns: term, estimate, std_error, statistic, p_value, s_value, conf_low, conf_high
