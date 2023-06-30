# Get Started

## Installation

Install the latest PyPi release:

``` python
pip install marginaleffects
```

## Estimands: Predictions, Comparisons, and Slopes

The `marginaleffects` package allows `python` users to compute and plot
three principal quantities of interest: (1) predictions, (2)
comparisons, and (3) slopes. In addition, the package includes a
convenience function to compute a fourth estimand, “marginal means”,
which is a special case of averaged predictions. `marginaleffects` can
also average (or “marginalize”) unit-level (or “conditional”) estimates
of all those quantities, and conduct hypothesis tests on them.

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

One ambiguous aspect of the definitions above is that the word
“marginal” comes up in two different and *opposite* ways:

1.  In “marginal effects,” we refer to the effect of a tiny (marginal)
    change in the regressor on the outcome. This is a slope, or
    derivative.
2.  In “marginal means,” we refer to the process of marginalizing across
    rows of a prediction grid. This is an average, or integral.

On this website and in this package, we reserve the expression “marginal
effect” to mean a “slope” or “partial derivative”.

The `marginaleffects` package includes functions to estimate, average,
plot, and summarize all of the estimands described above. The objects
produced by `marginaleffects` are “tidy”: they produce simple data
frames in “long” data frame format.

We now apply `marginaleffects` functions to compute each of the
estimands described above. First, we fit a linear regression model with
multiplicative interactions:

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
    Date:                Fri, 30 Jun 2023   Prob (F-statistic):           2.60e-10
    Time:                        15:50:51   Log-Likelihood:                -66.158
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

    polars.config.Config

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

    | rowid | estimate  | std_error | statistic | p_value    | … | qsec  | vs  | am  | gear | carb |
    |-------|-----------|-----------|-----------|------------|---|-------|-----|-----|------|------|
    | 0     | 22.488569 | 0.884149  | 25.43528  | 0.0        | … | 16.46 | 0   | 1   | 4    | 4    |
    | 1     | 20.801859 | 1.194205  | 17.419002 | 3.9968e-15 | … | 17.02 | 0   | 1   | 4    | 4    |
    | 2     | 25.264652 | 0.708531  | 35.657806 | 0.0        | … | 18.61 | 1   | 1   | 4    | 1    |
    | 3     | 20.255492 | 0.704464  | 28.753051 | 0.0        | … | 19.44 | 1   | 0   | 3    | 1    |
    | 4     | 16.997817 | 0.711866  | 23.877839 | 0.0        | … | 17.02 | 0   | 0   | 3    | 2    |

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

    | rowid | term | contra | estima | std_err | … | carb | marginalef | predict | predicted_ | predicted_ |
    |       |      | st     | te     | or      |   |      | fects_comp | ed      | lo         | hi         |
    |       |      |        |        |         |   |      | arison     |         |            |            |
    |-------|------|--------|--------|---------|---|------|------------|---------|------------|------------|
    | 0.0   | hp   | +1     | -0.036 | 0.01850 | … | 4.0  | difference | 22.4885 | 22.507022  | 22.470117  |
    |       |      |        | 906    | 2       |   |      |            | 69      |            |            |
    | 1.0   | hp   | +1     | -0.028 | 0.01562 | … | 4.0  | difference | 20.8018 | 20.816203  | 20.787514  |
    |       |      |        | 689    | 9       |   |      |            | 59      |            |            |
    | 2.0   | hp   | +1     | -0.046 | 0.02258 | … | 1.0  | difference | 25.2646 | 25.287938  | 25.241366  |
    |       |      |        | 572    | 7       |   |      |            | 52      |            |            |
    | 3.0   | hp   | +1     | -0.042 | 0.01328 | … | 1.0  | difference | 20.2554 | 20.276628  | 20.234357  |
    |       |      |        | 271    | 3       |   |      |            | 93      |            |            |
    | 4.0   | hp   | +1     | -0.039 | 0.01341 | … | 2.0  | difference | 16.9978 | 17.017326  | 16.978308  |
    |       |      |        | 018    | 1       |   |      |            | 17      |            |            |

The `comparisons()` function allows customized queries. For example,
what happens to the predicted outcome when the `hp` variable increases
from 100 to 120?

``` python
cmp = comparisons(mod, variables = {"hp": [120, 100]})
print(cmp)
```

    | rowid | term | contra | estima | std_err | … | carb | marginalef | predict | predicted_ | predicted_ |
    |       |      | st     | te     | or      |   |      | fects_comp | ed      | lo         | hi         |
    |       |      |        |        |         |   |      | arison     |         |            |            |
    |-------|------|--------|--------|---------|---|------|------------|---------|------------|------------|
    | 0.0   | hp   | 100 -  | 0.7381 | 0.37003 | … | 4.0  | difference | 22.4885 | 22.119514  | 22.857625  |
    |       |      | 120    | 11     | 4       |   |      |            | 69      |            |            |
    | 1.0   | hp   | 100 -  | 0.5737 | 0.31257 | … | 4.0  | difference | 20.8018 | 20.514965  | 21.088752  |
    |       |      | 120    | 87     | 2       |   |      |            | 59      |            |            |
    | 2.0   | hp   | 100 -  | 0.9314 | 0.45174 | … | 1.0  | difference | 25.2646 | 24.007217  | 24.93865   |
    |       |      | 120    | 33     | 3       |   |      |            | 52      |            |            |
    | 3.0   | hp   | 100 -  | 0.8454 | 0.26565 | … | 1.0  | difference | 20.2554 | 19.83278   | 20.678206  |
    |       |      | 120    | 26     | 6       |   |      |            | 93      |            |            |
    | …     | …    | …      | …      | …       | … | …    | …          | …       | …          | …          |
    | 28.0  | hp   | 100 -  | 0.3836 | 0.26979 | … | 4.0  | difference | 15.8961 | 18.658723  | 19.04241   |
    |       |      | 120    | 87     | 5       |   |      |            | 76      |            |            |
    | 29.0  | hp   | 100 -  | 0.6414 | 0.33446 | … | 6.0  | difference | 19.4116 | 21.175661  | 21.817111  |
    |       |      | 120    | 5      | 4       |   |      |            | 74      |            |            |
    | 30.0  | hp   | 100 -  | 0.1259 | 0.27216 | … | 8.0  | difference | 14.7881 | 16.141785  | 16.26771   |
    |       |      | 120    | 24     | 5       |   |      |            |         |            |            |
    | 31.0  | hp   | 100 -  | 0.6350 | 0.33226 | … | 2.0  | difference | 21.4619 | 21.112738  | 21.747744  |
    |       |      | 120    | 06     | 1       |   |      |            | 91      |            |            |

What happens to the predicted outcome when the `wt` variable increases
by 1 standard deviation about its mean?

``` python
cmp = comparisons(mod, variables = {"hp": "sd"})
print(cmp)
```

    | rowid | term | contra | estima | std_err | … | carb | marginalef | predict | predicted_ | predicted_ |
    |       |      | st     | te     | or      |   |      | fects_comp | ed      | lo         | hi         |
    |       |      |        |        |         |   |      | arison     |         |            |            |
    |-------|------|--------|--------|---------|---|------|------------|---------|------------|------------|
    | 0.0   | hp   | +68.56 | -2.530 | 1.26853 | … | 4.0  | difference | 22.4885 | 23.753745  | 21.223394  |
    |       |      | 286848 | 351    | 1       |   |      |            | 69      |            |            |
    |       |      | 932059 |        |         |   |      |            |         |            |            |
    | 1.0   | hp   | +68.56 | -1.967 | 1.07154 | … | 4.0  | difference | 20.8018 | 21.785371  | 19.818346  |
    |       |      | 286848 | 025    | 2       |   |      |            | 59      |            |            |
    |       |      | 932059 |        |         |   |      |            |         |            |            |
    | 2.0   | hp   | +68.56 | -3.193 | 1.54864 | … | 1.0  | difference | 25.2646 | 26.861196  | 23.668109  |
    |       |      | 286848 | 087    |         |   |      |            | 52      |            |            |
    |       |      | 932059 |        |         |   |      |            |         |            |            |
    | 3.0   | hp   | +68.56 | -2.898 | 0.91070 | … | 1.0  | difference | 20.2554 | 21.704613  | 18.806372  |
    |       |      | 286848 | 24     | 6       |   |      |            | 93      |            |            |
    |       |      | 932059 |        |         |   |      |            |         |            |            |
    | …     | …    | …      | …      | …       | … | …    | …          | …       | …          | …          |
    | 28.0  | hp   | +68.56 | -1.315 | 0.92489 | … | 4.0  | difference | 15.8961 | 16.553843  | 15.238509  |
    |       |      | 286848 | 334    | 5       |   |      |            | 76      |            |            |
    |       |      | 932059 |        |         |   |      |            |         |            |            |
    | 29.0  | hp   | +68.56 | -2.198 | 1.14658 | … | 6.0  | difference | 19.4116 | 20.511165  | 18.312183  |
    |       |      | 286848 | 983    | 9       |   |      |            | 74      |            |            |
    |       |      | 932059 |        |         |   |      |            |         |            |            |
    | 30.0  | hp   | +68.56 | -0.431 | 0.93302 | … | 8.0  | difference | 14.7881 | 15.003943  | 14.572257  |
    |       |      | 286848 | 686    | 1       |   |      |            |         |            |            |
    |       |      | 932059 |        |         |   |      |            |         |            |            |
    | 31.0  | hp   | +68.56 | -2.176 | 1.13903 | … | 2.0  | difference | 21.4619 | 22.550437  | 20.373546  |
    |       |      | 286848 | 891    | 7       |   |      |            | 91      |            |            |
    |       |      | 932059 |        |         |   |      |            |         |            |            |

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

    | term | contrast | estimate | std_error | statistic | p_value | s_value | conf_low | conf_high |
    |------|----------|----------|-----------|-----------|---------|---------|----------|-----------|
    | hp   | +50      | 0.909534 | 0.029058  | 31.300264 | 0.0     | inf     | 0.84956  | 0.969507  |

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
print(cmp)
```

    | term | contrast | estimate | std_error | statistic | p_value | s_value | conf_low | conf_high |
    |------|----------|----------|-----------|-----------|---------|---------|----------|-----------|
    | hp   | +50      | 0.909534 | 0.029058  | 31.300264 | 0.0     | inf     | 0.84956  | 0.969507  |

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

    | rowid | estimate   | std_err | statist | p_value   | … | drat     | qsec     | vs  | gear | carb |
    |       |            | or      | ic      |           |   |          |          |     |      |      |
    |-------|------------|---------|---------|-----------|---|----------|----------|-----|------|------|
    | 0     | 12.500384  | 1.95879 | 6.38166 | 0.000001  | … | 3.596562 | 17.84875 | 0   | 3    | 4    |
    |       |            | 5       | 9       |           |   |          |          |     |      |      |
    | 1     | 21.36604   | 2.40581 | 8.88101 | 4.7315e-9 | … | 3.596562 | 17.84875 | 0   | 3    | 4    |
    |       |            | 2       |         |           |   |          |          |     |      |      |
    | 2     | 7.414996   | 6.11720 | 1.21215 | 0.237252  | … | 3.596562 | 17.84875 | 0   | 3    | 4    |
    |       |            | 8       | 4       |           |   |          |          |     |      |      |
    | 3     | 25.093597  | 3.76836 | 6.65902 | 6.8999e-7 | … | 3.596562 | 17.84875 | 0   | 3    | 4    |
    |       |            | 1       | 2       |           |   |          |          |     |      |      |

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

    | estimate  | std_error | statistic | p_value | s_value | conf_low  | conf_high |
    |-----------|-----------|-----------|---------|---------|-----------|-----------|
    | 20.090625 | 0.390416  | 51.459497 | 0.0     | inf     | 19.284845 | 20.896405 |

This is equivalent to manual computation by:

``` python
np.mean(mod.predict())
```

    20.090625000000014

The main `marginaleffects` functions all include a `by` argument, which
allows us to marginalize within sub-groups of the data. For example,

``` python
cmp = avg_comparisons(mod, by = "am")
print(cmp)
```

    | am  | term | contra | estimate  | std_err | statist | p_value   | s_value   | conf_low | conf_hi |
    |     |      | st     |           | or      | ic      |           |           |          | gh      |
    |-----|------|--------|-----------|---------|---------|-----------|-----------|----------|---------|
    | 1.0 | am   | mean(1 | 1.902898  | 2.30862 | 0.82425 | 0.417912  | 1.258729  | -2.86187 | 6.66767 |
    |     |      | ) -    |           | 9       | 4       |           |           | 9        | 5       |
    |     |      | mean(0 |           |         |         |           |           |          |         |
    |     |      | )      |           |         |         |           |           |          |         |
    | 0.0 | am   | mean(1 | -1.383009 | 2.52499 | -0.5477 | 0.588937  | 0.763814  | -6.59434 | 3.82832 |
    |     |      | ) -    |           | 4       | 28      |           |           |          | 2       |
    |     |      | mean(0 |           |         |         |           |           |          |         |
    |     |      | )      |           |         |         |           |           |          |         |
    | 1.0 | wt   | +1     | -6.07176  | 1.97621 | -3.0724 | 0.005221  | 7.58135   | -10.1504 | -1.9930 |
    |     |      |        |           | 1       | 25      |           |           | 58       | 61      |
    | 0.0 | wt   | +1     | -2.479903 | 1.23162 | -2.0135 | 0.055405  | 4.173847  | -5.02186 | 0.06205 |
    |     |      |        |           | 9       | 14      |           |           |          | 5       |
    | 1.0 | hp   | +1     | -0.04364  | 0.02129 | -2.0497 | 0.051466  | 4.28024   | -0.08758 | 0.00030 |
    |     |      |        |           |         | 72      |           |           |          | 1       |
    | 0.0 | hp   | +1     | -0.034264 | 0.01586 | -2.1598 | 0.040994  | 4.608459  | -0.06700 | -0.0015 |
    |     |      |        |           | 4       | 28      |           |           | 5        | 22      |

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
hyp = hypotheses(mod, "b4 - 2. * b3 = 0")
print(hyp)
```

    | term       | estimate | std_error | statistic | p_value  | s_value  | conf_low   | conf_high |
    |------------|----------|-----------|-----------|----------|----------|------------|-----------|
    | b4-2.*b3=0 | 7.471607 | 37.603481 | 0.198695  | 0.843937 | 0.244792 | -69.555631 | 84.498846 |

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

    | rowid | term | contra | estima | std_err | … | carb | marginalef | predict | predicted_ | predicted_ |
    |       |      | st     | te     | or      |   |      | fects_comp | ed      | lo         | hi         |
    |       |      |        |        |         |   |      | arison     |         |            |            |
    |-------|------|--------|--------|---------|---|------|------------|---------|------------|------------|
    | 0.0   | drat | +1     | 10.241 | 5.16143 | … | 2.0  | difference | 25.7186 | 20.597943  | 30.839317  |
    |       |      |        | 374    | 2       |   |      |            | 3       |            |            |
    | 1.0   | drat | +1     | 5.2239 | 3.79106 | … | 2.0  | difference | 16.2756 | 13.663695  | 18.887621  |
    |       |      |        | 26     | 9       |   |      |            | 58      |            |            |

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

    | term  | estimate | std_error | statistic | p_value  | s_value  | conf_low   | conf_high |
    |-------|----------|-----------|-----------|----------|----------|------------|-----------|
    | b1=b2 | 5.017448 | 8.519298  | 0.588951  | 0.560616 | 0.834915 | -12.433542 | 22.468438 |
