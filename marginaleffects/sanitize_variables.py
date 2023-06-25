import re
import numpy as np
import polars as pl
from warnings import warn
from collections import namedtuple
HiLo = namedtuple('HiLo', ['variable', 'hi', 'lo', 'lab', "pad"])


def get_one_variable_type(variable, newdata):
    inttypes = [pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
    if variable not in newdata.columns:
        raise ValueError(f"`{variable}` is not in `newdata`")
    if newdata[variable].dtype in [pl.Utf8, pl.Categorical]:
        return "character"
    elif newdata[variable].dtype == pl.Boolean:
        return "boolean"
    elif newdata[variable].dtype in inttypes:
        if newdata[variable].is_in([0, 1]).all():
            return "binary"
        else:
            return "numeric"
    elif newdata[variable].dtype in [pl.Float32, pl.Float64]:
        return "numeric"
    else:
        raise ValueError(f"Unknown type for `{variable}`: {newdata[variable].dtype}")


def get_one_variable_hi_lo(variable, value, newdata, comparison, eps):
    msg = "`value` must be a numeric, a list of length two, or 'sd'"
    vartype = get_one_variable_type(variable, newdata)

    if value is None:
        if vartype == "character":
            value = "reference"
        else:
            if comparison in ["eyexavg", "dyexavg", "eydxavg", "dydxavg", "eyex", "dyex", "eydx", "dydx"]:
                value = eps
            else:
                value = 1

    if vartype == "boolean":
        out = HiLo(variable=variable, hi=pl.Series([True]), lo=pl.Series([False]), lab="True - False", pad = None)
        return [out]

    if vartype == "binary":
        out = HiLo(variable=variable, hi=pl.Series([1]), lo=pl.Series([0]), lab="1 - 0", pad = None)
        return [out]

    if vartype == "character":
        if isinstance(value, list) and len(value) == 2:
            out = HiLo(
                variable=variable,
                hi=pl.Series([value[1]]),
                lo=pl.Series([value[0]]),
                lab=f"{value[1]} - {value[0]}",
                pad = None)
            return [out]

        elif isinstance(value, str):
            out = get_categorical_combinations(variable, newdata[variable].unique().sort(), value)
            return out

        else:
            raise ValueError(msg)

    if vartype == "numeric" and isinstance(value, str):
        if value == "sd":
            value = np.std(newdata[variable])
            lab = "sd"
            hi = (newdata[variable] + value / 2).cast(newdata[variable].dtype)
            lo = (newdata[variable] - value / 2).cast(newdata[variable].dtype)
        else:
            raise ValueError(msg)

    elif isinstance(value, list):
        if len(value) != 2:
            raise ValueError(msg)
        lab = f"{value[1]} - {value[0]}"
        hi = pl.Series([value[1]])
        lo = pl.Series([value[0]])

    elif isinstance(value, (int, float)):
        lab = f"+{value}"
        hi = newdata[variable] + value / 2
        lo = newdata[variable] - value / 2

    else:
        raise ValueError(msg)

    if isinstance(value, list):
        lo = pl.Series([value[0]])
        hi = pl.Series([value[1]])
    else:
        lo = newdata[variable] - value / 2
        hi = newdata[variable] + value / 2

    out = [HiLo(variable=variable, lo=lo, hi=hi, lab=lab, pad = None)]
    return out


def get_variables_names(variables, model, newdata):
    if variables is None:
        variables = model.model.exog_names
        variables = [re.sub("\[.*\]", "", x) for x in variables]
        variables = [x for x in variables if x in newdata.columns]
        variables = pl.Series(variables).unique().to_list()
    elif isinstance(variables, str):
        variables = [variables]
    else:
        assert isinstance(variables, dict), "`variables` must be None, a dict, string, or list of strings"
    good = [x for x in variables if x in newdata.columns]
    bad = [x for x in variables if x not in newdata.columns]
    if len(bad) > 0:
        bad = ", ".join(bad)
        warn(f"Variable(s) not in newdata: {bad}")
    if len(good) == 0:
        raise ValueError("There is no valid column name in `variables`.")
    return variables


def sanitize_variables(variables, model, newdata, comparison, eps):
    out = []

    if variables is None:
        vlist = get_variables_names(variables, model, newdata)
        for v in vlist:
            out.append(get_one_variable_hi_lo(v, None, newdata, comparison, eps))

    elif isinstance(variables, dict):
        for v in variables:
            if v not in newdata.columns:
                del variables[v]
                warn(f"Variable {v} is not in newdata.")
            else:
                out.append(get_one_variable_hi_lo(v, variables[v], newdata, comparison, eps))

    elif isinstance(variables, str):
        if variables not in newdata.columns:
            raise ValueError(f"Variable {variables} is not in newdata.")
        out.append(get_one_variable_hi_lo(variables, None, newdata, comparison, eps))

    elif isinstance(variables, list):
        for v in variables:
            if v not in newdata.columns:
                warn(f"Variable {v} is not in newdata.")
            else:
                out.append(get_one_variable_hi_lo(v, None, newdata, comparison, eps))

    # unnest list of list of HiLo
    out = [item for sublist in out for item in sublist]

    return out
        


def get_categorical_combinations(variable, uniqs, combo="reference"):

    if not isinstance(combo, str):
        raise ValueError("The 'variables' value must be a string.")

    if len(uniqs) > 25:
        raise ValueError("There are too many unique categories to compute comparisons.")

    out = []

    if combo == "reference":
        for u in uniqs:
            if u != uniqs[0]:
                hl = HiLo(
                    variable=variable,
                    hi=pl.Series([u]),
                    lo=pl.Series([uniqs[0]]),
                    lab=f"{u} - {uniqs[0]}",
                    pad=uniqs)
                out.append(hl)
    elif combo == "revreference":
        last_element = uniqs[-1]
        for u in uniqs:
            if u != last_element:
                hl = HiLo(
                    variable=variable,
                    hi=pl.Series([u]),
                    lo=pl.Series([last_element]),
                    lab=f"{u} - {last_element}",
                    pad=uniqs)
                out.append(hl)
    elif combo == "sequential":
        for i in range(len(uniqs) - 1):
            hl = HiLo(
                variable=variable,
                hi=pl.Series([uniqs[i + 1]]),
                lo=pl.Series([uniqs[i]]),
                lab=f"{uniqs[i + 1]} - {uniqs[i]}",
                pad=uniqs)
            out.append(hl)
    elif combo == "revsequential":
        for i in range(len(uniqs) - 1, 0, -1):
            hl = HiLo(
                variable=variable,
                hi=pl.Series([uniqs[i - 1]]),
                lo=pl.Series([uniqs[i]]),
                lab=f"{uniqs[i - 1]} - {uniqs[i]}",
                pad=uniqs)
            out.append(hl)
    elif combo == "pairwise":
        for i in range(len(uniqs)):
            for j in range(i + 1, len(uniqs)):
                hl = HiLo(
                    variable=variable,
                    hi=pl.Series([uniqs[j]]),
                    lo=pl.Series([uniqs[i]]),
                    lab=f"{uniqs[j]} - {uniqs[i]}",
                    pad=uniqs)
                out.append(hl)
    elif combo == "revpairwise":
        for i in range(len(uniqs)):
            for j in range(i + 1, len(uniqs)):
                hl = HiLo(
                    variable=variable,
                    hi=pl.Series([uniqs[i]]),
                    lo=pl.Series([uniqs[j]]),
                    lab=f"{uniqs[i]} - {uniqs[j]}",
                    pad=uniqs)
                out.append(hl)
    else:
        raise ValueError(f"The supported comparisons are: 'reference', 'revreference', 'sequential', 'revsequential', 'pairwise', and 'revpairwise'.")

    return out