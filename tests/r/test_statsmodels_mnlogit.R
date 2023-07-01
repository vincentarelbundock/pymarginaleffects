source(here::here("tests/r/load.R"))

dat = fread("https://vincentarelbundock.github.io/Rdatasets/csv/palmerpenguins/penguins.csv")
mod = multinom(island ~ bill_length_mm + flipper_length_mm, data = dat, trace = FALSE)

predictions(mod, type = "probs") |> 
    fwrite(here("tests/r/test_statsmodels_mnlogit_predictions_01.csv"))

predictions(mod, by = "island", type = "probs") |> 
    fwrite(here("tests/r/test_statsmodels_mnlogit_predictions_02.csv"))

comparisons(mod, type = "probs") |>
    fwrite(here("tests/r/test_statsmodels_mnlogit_comparisons_01.csv"))

comparisons(mod, by = "island", type = "probs") |>
    fwrite(here("tests/r/test_statsmodels_mnlogit_comparisons_02.csv"))
