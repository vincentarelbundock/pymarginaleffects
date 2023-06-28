library(palmerpenguins)
library(marginaleffects)
library(nnet)
dat = penguins
dat$island <- relevel(dat$island, ref = "Torgersen")
mod = multinom(island ~ bill_length_mm + flipper_length_mm, dat, hess = TRUE)
predictions(mod) |> head()

predictions(mod) |> attr("jacobian") |> head()
