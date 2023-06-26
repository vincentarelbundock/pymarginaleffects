library(dplyr)
library(marginaleffects)
mod = lm(mpg ~ wt * hp * cyl, mtcars)
comparisons(mod) |> arrange(term, contrast, rowid)
