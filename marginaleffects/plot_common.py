import numpy as np
from .datagrid import datagrid  # noqa
from .sanitize_model import sanitize_model
import polars as pl


def dt_on_condition(model, condition):
    model = sanitize_model(model)

    condition_new = condition  # two pointers to the same object? this looks like a bug

    # not sure why newdata gets added
    modeldata = model.modeldata

    if isinstance(condition_new, str):
        condition_new = [condition_new]

    to_datagrid = {}
    first_key = ""  # special case when the first element is numeric

    if isinstance(condition_new, list):
        assert all(
            ele in modeldata.columns for ele in condition_new
        ), "All elements of condition must be columns of the model."
        first_key = condition_new[0]
        to_datagrid = {key: None for key in condition_new}

    elif isinstance(condition_new, dict):
        assert all(
            key in modeldata.columns for key in condition_new.keys()
        ), "All keys of condition must be columns of the model."
        first_key = next(iter(condition_new))
        to_datagrid = (
            condition_new  # third pointer to the same object? looks like a BUG
        )

    # not sure why `newdata` sometimes gets added
    if isinstance(condition_new, dict) and "newdata" in to_datagrid.keys():
        condition_new.pop("newdata", None)

    assert (
        1 <= len(condition_new) <= 4
    ), f"Lenght of condition must be inclusively between 1 and 4. Got : {len(condition_new)}."

    for key, value in to_datagrid.items():
        variable_type = model.variables_type[key]

        if variable_type in ["numeric", "integer"]:
            to_datagrid[key] = condition_numeric(
                modeldata, key, value, key == first_key
            )

        elif variable_type in ["boolean", "character", "binary"]:
            # get specified names of the condition
            # here is the BUG, we take the values of "species" back from the model
            to_datagrid[key] = (
                to_datagrid[key]
                if to_datagrid[key]
                else modeldata[key].unique().sort().to_list()
            )
            assert (
                len(to_datagrid[key]) <= 10
            ), f"Character type variables of more than 10 unique values are not supported. {key} variable has {len(to_datagrid[key])} unique values."

    to_datagrid["newdata"] = modeldata
    dt = datagrid(**to_datagrid)
    return dt  # noqa: F821


def condition_numeric(modeldata, key, value, first):
    if value is None:
        if first:
            out = np.linspace(modeldata[key].min(), modeldata[key].max(), 100).tolist()
        else:
            out = np.percentile(
                modeldata[key], [0, 25, 50, 75, 100], method="midpoint"
            ).tolist()
    elif isinstance(value, str) and value == "threenum":
        m = modeldata[key].mean()
        s = modeldata[key].std()
        out = [m - s, m, m + s]
    elif isinstance(value, str) and value == "fivenum":
        out = [0, 0.25, 0.5, 0.75, 1]
        out = [modeldata[key].quantile(x) for x in out]
    elif isinstance(value, str) and value == "minmax":
        out = [0, 1]
        out = [modeldata[key].quantile(x) for x in out]
    else:
        out = value

    return out


def plot_labels(model, dt, condition):
    if not isinstance(condition, dict):
        return dt

    for k, v in condition.items():
        if model.variables_type[k] in ["numeric", "integer"]:
            if condition[k] == "threenum":
                lab = ["-SD", "Mean", "+SD"]
                dt = ordered_cat(dt, k, lab)
            elif condition[k] == "fivenum":
                lab = ["Min", "Q1", "Q2", "Q3", "Max"]
                dt = ordered_cat(dt, k, lab)
            elif condition[k] == "minmax":
                lab = ["Min", "Max"]
                dt = ordered_cat(dt, k, lab)
    return dt


# polars does not seem to have a custom ordered categorical. only physical and lexical.
def ordered_cat(dt, k, lab):
    uniq = dict(zip(dt[k].unique().sort(), list(range(len(lab)))))
    dt = dt.with_columns(dt[k].replace_strict(uniq).alias(k))
    dt = dt.sort(by=k)
    uniq = dict(zip(list(range(len(lab))), lab))
    dt = dt.with_columns(dt[k].replace_strict(uniq).cast(pl.Categorical).alias(k))
    dt = dt.sort(by="rowid")
    return dt
