import numpy as np
import polars as pl
from warnings import warn

def get_one_variable_type(variable, newdata):
    floattypes = [pl.Float32, pl.Float64]
    inttypes = [pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
    numtypes = floattypes + inttypes
    if variable not in newdata.columns:
        raise ValueError(f"`{variable}` is not in `newdata`")
    if newdata[variable].dtype == pl.Utf8:
        return "character"
    elif newdata[variable].dtype == pl.Boolean:
        return "binary"
    elif newdata[variable].dtype in inttypes and newdata[variable].is_in([0, 1]).all():
        return "binary"
    elif newdata[variable].dtype in numtypes:
        return "numeric"
    else:
        raise ValueError(f"Unknown type for `{variable}`: {newdata[variable].dtype}")


def get_one_variable_hi_lo(variable, value, newdata):
    msg = "`value` must be a numeric, a list of length two, or 'sd'"
    vartype = get_one_variable_type(variable, newdata)
    if value is None:
        value = 1
    if vartype == "binary":
        out = dict(hi = pl.Series([1]), lo = pl.Series([0]), lab = "1 - 0")
        return out
    if vartype == "character":
        if isinstance(value, list) or len(value) != 2:
            raise ValueError("For character variables, `value` must be a list of length two.")
        out = dict(hi = pl.Series([value[1]]), lo = pl.Series([value[0]]), lab = f"{value[1]} - {value[0]}")
    if isinstance(value, str):
        if value == "sd":
            value = np.std(newdata[variable])
            lab = "sd"
            hi = newdata[variable] + value / 2
            lo = newdata[variable] - value / 2
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
    out = dict(variable = variable, lo=lo, hi=hi, lab=lab)
    return out 


def get_variables_names(variables, fit, newdata):
    if variables is None:
        variables = fit.model.exog_names
        variables = [x for x in variables if x in newdata.columns]
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


def sanitize_variables(variables, fit, newdata):
    out = {}
    
    if variables is None:
        vlist = get_variables_names(variables, fit, newdata)
        for v in vlist:
            out[v] = get_one_variable_hi_lo(v, None, newdata)

    elif isinstance(variables, dict):
        for v in variables:
            if v not in newdata.columns:
                del variables[v]
                warn(f"Variable {v} is not in newdata.")
            out[v] = get_one_variable_hi_lo(v, variables[v], newdata)

    elif isinstance(variables, str):
        if variables not in newdata.columns:
            raise ValueError(f"Variable {variables} is not in newdata.")
        out[variables] = get_one_variable_hi_lo(variables, None, newdata)

    elif isinstance(variables, list):
        for v in variables:
            if v not in variables:
                warn(f"Variable {v} is not in newdata.")
            else:
                out[v] = get_one_variable_hi_lo(v, None, newdata)

    return out

        


