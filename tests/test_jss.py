import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from marginaleffects.classes import MarginaleffectsDataFrame
from .utilities import *

dat = pl.read_csv("tests/data/impartiality.csv") \
  .with_columns(pl.col("impartial").cast(pl.Int8))

m = smf.logit(
  "impartial ~ equal * democracy + continent",
  data = dat.to_pandas()
).fit()


def test_predictions():

  p = predictions(m)
  assert isinstance(p, pl.DataFrame)

  p = predictions(m, newdata = dat.head())
  assert isinstance(p, pl.DataFrame)

  p = predictions(m, newdata = "mean")
  assert isinstance(p, pl.DataFrame)

  p = predictions(m,
    newdata = datagrid(model = m, democracy = dat["democracy"].unique(), equal = [30, 90])
  )
  assert isinstance(p, pl.DataFrame)
  assert p.shape[0] == 4

  p1 = avg_predictions(m)
  p2 = np.mean(m.predict(dat.to_pandas()).to_numpy())
  assert isinstance(p1, pl.DataFrame)
  assert p1.shape[0] == 1
  assert p1["estimate"][0] == p2

  p = predictions(m, by = "democracy")
  assert isinstance(p, pl.DataFrame)
  assert p.shape[0] == 2

  p = plot_predictions(m, by = ["democracy", "continent"])
  assert assert_image(p, label = "jss_01", file = "jss") is None



# ## -----------------------------------------------------------------------------
# coef(m)[c("continentAsia", "continentAmericas")]

# hypotheses(m, hypothesis = "continentAsia = continentAmericas")


# ## -----------------------------------------------------------------------------
# avg_predictions(m,
#   by = "democracy",
#   type = "response")


# ## -----------------------------------------------------------------------------
# avg_predictions(m,
#   by = "democracy",
#   type = "response",
#   hypothesis = "revpairwise")


# ## -----------------------------------------------------------------------------
# predictions(m,
#   by = "democracy",
#   type = "response",
#   hypothesis = "b2 = b1 * 2")


# ## ----include=FALSE------------------------------------------------------------
# tmp = avg_predictions(m, by = "democracy", type = "response") 
# res = sprintf("%.4f", tmp$estimate[2] - 2 * tmp$estimate[1])
# tmp = transform(tmp, estimate = sprintf("%.4f", estimate))


# ## -----------------------------------------------------------------------------
# predictions(m,
#   by = "democracy",
#   type = "response",
#   hypothesis = "b2 = b1 * 2",
#   equivalence = c(-.2, .2))


# ## ----comparisons-regime-original-data-----------------------------------------
# comparisons(m, variables = "democracy")


# ## ----comparisons-regime-avg---------------------------------------------------
# avg_comparisons(m)


# ## -----------------------------------------------------------------------------
# dat_lo = transform(dat, democracy = "Autocracy")
# dat_hi = transform(dat, democracy = "Democracy")
# pred_lo = predict(m, newdata = dat_lo, type = "response")
# pred_hi = predict(m, newdata = dat_hi, type = "response")
# mean(pred_hi - pred_lo)


# ## ----eval = FALSE-------------------------------------------------------------
# ## avg_comparisons(m, variables = list("equal" = 4))
# ## avg_comparisons(m, variables = list("equal" = "sd"))
# ## avg_comparisons(m, variables = list("equal" = "iqr"))
# ## avg_comparisons(m, variables = list("equal" = c(30, 90)))


# ## -----------------------------------------------------------------------------
# avg_comparisons(m, variables = "democracy", comparison = "ratio")


# ## -----------------------------------------------------------------------------
# avg_comparisons(m,
#   comparison = "lnor",
#   transform = exp)


# ## -----------------------------------------------------------------------------
# avg_comparisons(m,
#   variables = "equal",
#   comparison = \(hi, lo) mean(hi) / mean(lo))


# ## ----include=FALSE------------------------------------------------------------
# cmp = comparisons(m,
#   by = "democracy",
#   variables = list(equal = c(30, 90)))


# ## -----------------------------------------------------------------------------
# cmp = comparisons(m,
#   by = "democracy",
#   variables = list(equal = c(30, 90)))
# cmp


# ## ----fig.cap = "Effect of a change from 30 to 90 in resource equality on the predicted probability of having impartial public institutions.\\label{fig:comparisons-democracy}"----
# plot_comparisons(m,
#   by = "democracy",
#   variables = list(equal = c(30, 90))) +
#   labs(x = NULL, y = "Risk Difference (Impartiality)")


# ## -----------------------------------------------------------------------------
# cmp = comparisons(m,
#   by = "democracy",
#   variables = list(equal = c(30, 90)),
#   hypothesis = "pairwise")
# cmp


# ## ----tangents, echo=FALSE, fig.cap = "Tangents to the prediction function at 25 and 50.", fig.pos="h"----
# p = predictions(m, datagrid(equal = c(25, 50)))
# s = slopes(m, datagrid(equal = c(25, 50)), variables = "equal")
# tan1 = data.frame(x = c(25, 50), y = c(0.726, 0.726 + 0.01667 * (50 - 25)))
# tan2 = data.frame(x = c(25, 75), y = c(0.956, 0.956 + 0.00355 * (75 - 50)))
# plot_predictions(m, condition = "equal", vcov = FALSE) +
#   geom_abline(slope = s$estimate[1], intercept = p$estimate[1] - 25 * s$estimate[1], color = "red", linetype = 2) +
#   geom_abline(slope = s$estimate[2], intercept = p$estimate[2] - 50 * s$estimate[2], color = "red", linetype = 2) +
#   theme_bw()


# ## -----------------------------------------------------------------------------
# slopes(m, newdata = datagrid(equal = c(25, 50)), variables = "equal")


# ## -----------------------------------------------------------------------------
# avg_slopes(m, variables = "equal")


# ## -----------------------------------------------------------------------------
# slopes(m, variables = "equal", newdata = "mean")


# ## -----------------------------------------------------------------------------
# slopes(m, variables = "equal", newdata = "median", by = "democracy")


# ## -----------------------------------------------------------------------------
# avg_slopes(m, variables = "equal", slope = "eyex")


tit = pl.read_csv("tests/data/titanic.csv")
mod_tit = smf.ols("Survived ~ Woman * Passenger_Class", data = tit.to_pandas()).fit()

avg_predictions(mod_tit,
    newdata = datagrid(
      Passenger_Class = tit["Passenger_Class"].unique(),
      Woman = tit["Woman"].unique(),
      model = mod_tit),
    by = "Woman")


avg_predictions(mod_tit,
    newdata = datagrid(
      Passenger_Class = tit["Passenger_Class"].unique(),
      Woman = tit["Woman"].unique(),
      model = mod_tit),
    by = "Woman",
    hypothesis = "revpairwise")


avg_comparisons(mod_tit,
    variables = "Woman",
    newdata = datagrid(
      Passenger_Class = tit["Passenger_Class"].unique(),
      Woman = tit["Woman"].unique(),
      model = mod_tit))


tit.group_by("Passenger_Class").count()


avg_comparisons(mod_tit, variables = "Woman")


# Risk difference by passenger class
avg_comparisons(mod_tit,
    variables = "Woman",
    by = "Passenger_Class")


avg_comparisons(mod_tit,
    variables = "Woman",
    by = "Passenger_Class",
    hypothesis = "b1 - b3 = 0")



def test_python_section():
  p = avg_predictions(m, by="continent")
  assert isinstance(p, pl.DataFrame)
  assert p.shape[0] == 4

  s = slopes(m, newdata="mean")
  assert isinstance(s, pl.DataFrame)
  assert s.shape[0] == 5
