import polars as pl
import statsmodels.formula.api as smf
from marginaleffects import *
from plotnine import *


tit = pl.read_csv("tests/data/titanic.csv")
mod_tit = smf.ols("Survived ~ Woman * Passenger_Class", data = tit.to_pandas()).fit()

# Risk difference by passenger class
p = plot_predictions(mod_tit, condition = "Woman")
ggsave(p, filename = "trash_01.png")
p = plot_predictions(mod_tit, condition = ["Woman", "Passenger_Class"])
ggsave(p, filename = "trash_02.png")


# LetsPlot.setup_html()
# z = (
# ggplot(p, aes(x = "Woman", y = "estimate", ymin = "conf_low", ymax = "conf_high", color = "Passenger_Class")) +
#     geom_linerange() +
#     labs(x = "Blah blah", y = "Estimate") +
#     facet_wrap("Passenger_Class")
# )
# ggsave(z, "trash.png")

