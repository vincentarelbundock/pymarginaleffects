library(marginaleffects)
df = read.csv("https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Guerry.csv")
mod = lm(Literacy ~ Pop1831 * Desertion, df)
comparisons(mod, variables = "Pop1831")

predictions(mod) |> head()
