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
    Date:                Mon, 24 Jul 2023   Prob (F-statistic):           2.60e-10
    Time:                        18:11:21   Log-Likelihood:                -66.158
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

    | Estimate | Std.Error | z    | P(>|z|) | S    | [    | ]    |
    |----------|-----------|------|---------|------|------|------|
    | 22.5     | 0.884     | 25.4 | 0       | inf  | 20.7 | 24.3 |
    | 20.8     | 1.19      | 17.4 | 4e-15   | 47.8 | 18.3 | 23.3 |
    | 25.3     | 0.709     | 35.7 | 0       | inf  | 23.8 | 26.7 |
    | 20.3     | 0.704     | 28.8 | 0       | inf  | 18.8 | 21.7 |
    | 17       | 0.712     | 23.9 | 0       | inf  | 15.5 | 18.5 |

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

    | Term | Contrast | Estimate | Std.Error | … | P(>|z|) | S    | [       | ]       |
    |------|----------|----------|-----------|---|---------|------|---------|---------|
    | hp   | +1       | -0.0369  | 0.0185    | … | 0.0575  | 4.12 | -0.0751 | 0.00128 |
    | hp   | +1       | -0.0287  | 0.0156    | … | 0.0788  | 3.67 | -0.0609 | 0.00357 |
    | hp   | +1       | -0.0466  | 0.0226    | … | 0.0502  | 4.32 | -0.0932 | 4.6e-05 |
    | hp   | +1       | -0.0423  | 0.0133    | … | 0.00401 | 7.96 | -0.0697 | -0.0149 |
    | hp   | +1       | -0.039   | 0.0134    | … | 0.00769 | 7.02 | -0.0667 | -0.0113 |

    Columns: rowid, term, contrast, estimate, std_error, statistic, p_value, s_value, conf_low, conf_high, predicted, predicted_lo, predicted_hi, , mpg, cyl, disp, hp, drat, wt, qsec, vs, am, gear, carb

The `comparisons()` function allows customized queries. For example,
what happens to the predicted outcome when the `hp` variable increases
from 100 to 120?

``` python
cmp = comparisons(mod, variables = {"hp": [120, 100]})
print(cmp)
```

    | Term | Contrast  | Estimate | Std.Error | … | P(>|z|) | S     | 2.5%      | 97.5% |
    |------|-----------|----------|-----------|---|---------|-------|-----------|-------|
    | hp   | 100 - 120 | 0.738    | 0.37      | … | 0.0576  | 4.12  | -0.0256   | 1.5   |
    | hp   | 100 - 120 | 0.574    | 0.313     | … | 0.0788  | 3.67  | -0.0713   | 1.22  |
    | hp   | 100 - 120 | 0.931    | 0.452     | … | 0.0502  | 4.32  | -0.000918 | 1.86  |
    | hp   | 100 - 120 | 0.845    | 0.266     | … | 0.00401 | 7.96  | 0.297     | 1.39  |
    | …    | …         | …        | …         | … | …       | …     | …         | …     |
    | hp   | 100 - 120 | 0.384    | 0.27      | … | 0.168   | 2.57  | -0.173    | 0.941 |
    | hp   | 100 - 120 | 0.641    | 0.334     | … | 0.0671  | 3.9   | -0.0488   | 1.33  |
    | hp   | 100 - 120 | 0.126    | 0.272     | … | 0.648   | 0.626 | -0.436    | 0.688 |
    | hp   | 100 - 120 | 0.635    | 0.332     | … | 0.068   | 3.88  | -0.0507   | 1.32  |

    Columns: rowid, term, contrast, estimate, std_error, statistic, p_value, s_value, conf_low, conf_high, predicted, predicted_lo, predicted_hi, , mpg, cyl, disp, hp, drat, wt, qsec, vs, am, gear, carb

What happens to the predicted outcome when the `wt` variable increases
by 1 standard deviation about its mean?

``` python
cmp = comparisons(mod, variables = {"hp": "sd"})
print(cmp)
```

    | Term | Contrast           | Estimate | Std.Error | … | P(>|z|) | S     | 2.5%  | 97.5%   |
    |------|--------------------|----------|-----------|---|---------|-------|-------|---------|
    | hp   | +68.56286848932059 | -2.53    | 1.27      | … | 0.0576  | 4.12  | -5.15 | 0.0878  |
    | hp   | +68.56286848932059 | -1.97    | 1.07      | … | 0.0788  | 3.67  | -4.18 | 0.245   |
    | hp   | +68.56286848932059 | -3.19    | 1.55      | … | 0.0502  | 4.32  | -6.39 | 0.00315 |
    | hp   | +68.56286848932059 | -2.9     | 0.911     | … | 0.00401 | 7.96  | -4.78 | -1.02   |
    | …    | …                  | …        | …         | … | …       | …     | …     | …       |
    | hp   | +68.56286848932059 | -1.32    | 0.925     | … | 0.168   | 2.57  | -3.22 | 0.594   |
    | hp   | +68.56286848932059 | -2.2     | 1.15      | … | 0.0671  | 3.9   | -4.57 | 0.167   |
    | hp   | +68.56286848932059 | -0.432   | 0.933     | … | 0.648   | 0.626 | -2.36 | 1.49    |
    | hp   | +68.56286848932059 | -2.18    | 1.14      | … | 0.068   | 3.88  | -4.53 | 0.174   |

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

    | Term | Contrast | Estimate | Std.Error | … | P(>|z|) | S   | 2.5% | 97.5% |
    |------|----------|----------|-----------|---|---------|-----|------|-------|
    | hp   | +50      | 0.91     | 0.0291    | … | 0       | inf | 0.85 | 0.97  |

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

    | Term | Contrast | Estimate | Std.Error | … | P(>|z|)  | S    | 2.5%   | 97.5% |
    |------|----------|----------|-----------|---|----------|------|--------|-------|
    | mpg  | dY/dX    | 0.0665   | 0.0178    | … | 0.000798 | 10.3 | 0.0301 | 0.103 |

    Columns: term, contrast, estimate, std_error, statistic, p_value, s_value, conf_low, conf_high

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

    | Term | Contrast | Estimate | Std.Error | … | P(>|z|) | S    | 2.5%   | 97.5% |
    |------|----------|----------|-----------|---|---------|------|--------|-------|
    | mpg  | dY/dX    | 0.0732   | 0.0283    | … | 0.0148  | 6.08 | 0.0154 | 0.131 |

    Columns: term, contrast, estimate, std_error, statistic, p_value, s_value, conf_low, conf_high

``` python
mfx = slopes(mod, newdata = "median")
print(mfx)
```

    | Term | Contrast | Estimate | Std.Error | … | P(>|z|) | S    | 2.5%   | 97.5% |
    |------|----------|----------|-----------|---|---------|------|--------|-------|
    | mpg  | dY/dX    | 0.0679   | 0.0253    | … | 0.0118  | 6.41 | 0.0162 | 0.12  |

    Columns: term, contrast, estimate, std_error, statistic, p_value, s_value, conf_low, conf_high

We can also compute an “average slope” or “average marginaleffects”

``` python
mfx = avg_slopes(mod)
print(mfx)
```

    | Term | Contrast    | Estimate | Std.Error | … | P(>|z|)  | S    | 2.5%   | 97.5%  |
    |------|-------------|----------|-----------|---|----------|------|--------|--------|
    | mpg  | mean(dY/dX) | 0.0465   | 0.00886   | … | 1.16e-05 | 16.4 | 0.0284 | 0.0646 |

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

    | Estimate | Std.Error | z    | P(>|z|)  | S    | 2.5%    | 97.5% |
    |----------|-----------|------|----------|------|---------|-------|
    | 0.119    | 0.0778    | 1.53 | 0.136    | 2.88 | -0.0396 | 0.278 |
    | 0.492    | 0.12      | 4.11 | 0.000281 | 11.8 | 0.247   | 0.736 |

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

    | Estimate | Std.Error | z    | P(>|z|) | S    | 2.5%  | 97.5% |
    |----------|-----------|------|---------|------|-------|-------|
    | 0.393    | 0.108     | 3.63 | 0.00106 | 9.89 | 0.172 | 0.614 |
    | 0.393    | 0.108     | 3.63 | 0.00106 | 9.89 | 0.172 | 0.614 |
    | 0.393    | 0.108     | 3.63 | 0.00106 | 9.89 | 0.172 | 0.614 |
    | 0.393    | 0.108     | 3.63 | 0.00106 | 9.89 | 0.172 | 0.614 |

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

    | Estimate | Std.Error | z    | P(>|z|)  | S    | 2.5%  | 97.5% |
    |----------|-----------|------|----------|------|-------|-------|
    | 0.406    | 0.0688    | 5.91 | 1.81e-06 | 19.1 | 0.266 | 0.547 |

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

    | am | Term | Contrast | Estimate | … | P(>|z|)  | S    | 2.5%   | 97.5%  |
    |----|------|----------|----------|---|----------|------|--------|--------|
    | 1  | mpg  | +1       | 0.0449   | … | 1.44e-07 | 22.7 | 0.0315 | 0.0584 |
    | 0  | mpg  | +1       | 0.0475   | … | 0.000285 | 11.8 | 0.0239 | 0.0711 |

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

    | Term | Contrast                 | Estimate | Std.Error | … | P(>|z|) | S    | 2.5%    | 97.5%   |
    |------|--------------------------|----------|-----------|---|---------|------|---------|---------|
    | hp   | +1                       | -0.0442  | 0.0146    | … | 0.00527 | 7.57 | -0.0742 | -0.0143 |
    | cyl  | 6 - 4                    | -3.92    | 1.54      | … | 0.0167  | 5.91 | -7.08   | -0.77   |
    | cyl  | 8 - 4                    | -3.53    | 2.5       | … | 0.169   | 2.56 | -8.67   | 1.6     |
    | am   | mean(True) - mean(False) | 4.16     | 1.26      | … | 0.00266 | 8.55 | 1.58    | 6.74    |

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

    | Term | Contrast | Estimate | Std.Error | … | P(>|z|) | S    | 2.5%   | 97.5% |
    |------|----------|----------|-----------|---|---------|------|--------|-------|
    | drat | +1       | 10.2     | 5.16      | … | 0.0571  | 4.13 | -0.331 | 20.8  |
    | drat | +1       | 5.22     | 3.79      | … | 0.179   | 2.48 | -2.54  | 13    |

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

    | Term  | Estimate | Std.Error | z     | P(>|z|) | S     | 2.5%  | 97.5% |
    |-------|----------|-----------|-------|---------|-------|-------|-------|
    | b1=b2 | 5.02     | 8.52      | 0.589 | 0.561   | 0.835 | -12.4 | 22.5  |

    Columns: term, estimate, std_error, statistic, p_value, s_value, conf_low, conf_high
