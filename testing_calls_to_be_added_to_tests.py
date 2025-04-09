####################
# ADD TO TESTS AND
# REMOVE BEFORE MERGING
####################
#%%
from marginaleffects import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
from plotnine import *
import pandas as pd
import polars as pl

experiment = "not_discrete_interval_len2"
#%%
if experiment == "thornton":
    data = get_dataset("thornton", package = "marginaleffects")
    print(type(data))

    data = data.drop_nulls()
    mod = smf.logit("outcome ~ agecat + incentive", data=data).fit()
    fig = plot_predictions(mod, condition = ["agecat", "incentive"], gray = True)
    fig.show()

elif experiment == "not_discrete_interval_len2":
    data = get_dataset("mtcars", "datasets")

    data = data.drop_nulls()
    mod = smf.glm("mpg ~ hp * wt * cyl * gear", data=data).fit()

    fig = plot_predictions(mod, condition = ["gear", "hp", "cyl"], gray = True)
    fig.show() 

elif experiment == "discrete_interval_len1": # this gets inside discrete branch of plot_common()
    data = (
    get_dataset("mtcars", "datasets")
    .sort("gear")
    .with_columns(pl.col("gear").cast(pl.String).cast(pl.Categorical))
    .to_pandas()
    )

    mod = smf.ols("mpg ~ wt + C(gear)", data=data).fit()

    fig = plot_predictions(mod, condition = ["gear"], gray = True)
    fig.show() 
elif experiment == "discrete_interval_len2": # this gets inside discrete branch of plot_common()
    data = (
    get_dataset("mtcars", "datasets")
    .sort("gear")
    .with_columns(pl.col("gear").cast(pl.String).cast(pl.Categorical))
    .to_pandas()
    )

    mod = smf.ols("mpg ~ wt + C(gear)", data=data).fit()

    fig = plot_predictions(mod, condition = ["gear", "hp", "cyl"], gray = True)
    fig.show() 
elif experiment == "discrete_not_interval":
    data = (
    get_dataset("mtcars", "datasets")
    .sort("gear")
    .with_columns(pl.col("gear").cast(pl.String).cast(pl.Categorical))
    .to_pandas()
    )
    mod = smf.ols("mpg ~ wt + C(gear)", data=data).fit()

    fig = plot_predictions(mod, condition = ["gear"], vcov = False, gray = True)
    fig.show() 

elif experiment == "mtcars_categorical_x_not_interval": # this gets inside discrete and NOT interval branch of plot_common()
    data = get_dataset("mtcars", "datasets")

    data = data.drop_nulls()
    mod = smf.glm("mpg ~ hp * wt * cyl * gear", data=data).fit()

    fig = plot_predictions(mod, condition = ["gear", "hp", "cyl"], gray = True, vcov = False)
    fig.show() 
