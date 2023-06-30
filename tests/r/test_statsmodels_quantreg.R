source(here::here("tests/r/load.R"))

dat = iris[!is.na(iris$Species),]
mod = rq(Sepal.Length ~ Sepal.Width * Petal.Width + Species, tau = .25, data = dat)

predictions(mod, by = "Species")
slopes(mod, by = "Species")

predictions(mod) |> 
    fwrite(here("tests/r/test_statsmodels_quantreg_predictions_01.csv"))

predictions(mod, by = "Species") |> 
    fwrite(here("tests/r/test_statsmodels_quantreg_predictions_02.csv"))

comparisons(mod) |>
    fwrite(here("tests/r/test_statsmodels_quantreg_comparisons_01.csv"))

comparisons(mod, by = "Species") |>
    fwrite(here("tests/r/test_statsmodels_quantreg_comparisons_02.csv"))
