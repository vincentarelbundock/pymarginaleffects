import re
from collections import namedtuple
from warnings import warn

import numpy as np
import polars as pl
import pandas as pd

from .datagrid import datagrid
from .estimands import estimands
from .utils import get_modeldata, get_variable_type


def sanitize_vcov(vcov, model):
    if isinstance(vcov, bool):
        if vcov is True:
            V = model.cov_params()
        else:
            V = None
    elif isinstance(vcov, str):
        lab = f"cov_{vcov}"
        if hasattr(model, lab):
            V = getattr(model, lab)
        else:
            raise ValueError(f"The model object has no {lab} attribute.")
    else:
        raise ValueError(
            '`vcov` must be a boolean or a string like "HC3", which corresponds to an attribute of the model object such as "vcov_HC3".'
        )
    # mnlogit returns pandas
    try:
        V = V.to_numpy()
    except:
        pass
    return V


def sanitize_by(by):
    if by is True:
        by = ["group"]
    elif isinstance(by, str):
        by = ["group", by]
    elif isinstance(by, list):
        by = ["group"] + by
    elif by is False:
        by = False
    else:
        raise ValueError("The `by` argument must be True, False, a string, or a list of strings.")
    return by


def sanitize_newdata(model, newdata, wts, by = []):
    modeldata = get_modeldata(model)

    if newdata is None:
        out = modeldata

    elif isinstance(newdata, str) and newdata == "mean":
        out = datagrid(newdata=modeldata)

    elif isinstance(newdata, str) and newdata == "median":
        out = datagrid(newdata=modeldata, FUN_numeric=lambda x: x.median())

    elif isinstance(newdata, pd.DataFrame):
        out = pl.from_pandas(newdata)
        
    else:
        out = newdata

    if isinstance(by, list) and len(by) > 0:
        by = [x for x in by if x in out.columns]
        if len(by) > 0:
            out = out.sort(by)

    out = out.with_columns(
        pl.Series(range(out.height), dtype=pl.Int32).alias("rowid")
    )

    if wts is not None:
        if (isinstance(wts, str) is False) or (wts not in out.columns):
            raise ValueError(f"`newdata` does not have a column named '{wts}'.")

    xnames = get_variables_names(variables=None, model=model, newdata=out)
    ynames = model.model.data.ynames
    if isinstance(ynames, str):
        ynames = [ynames]
    cols = [x for x in xnames + ynames if x in out.columns]
    out = out.drop_nulls(subset=cols)

    if any([isinstance(out[x], pl.Categorical) for x in out.columns]):
        raise ValueError("Categorical type columns are not supported in `newdata`.")

    return out


def sanitize_comparison(comparison, by, wts=None):
    out = comparison
    if by is not False:
        if f"{comparison}avg" in estimands.keys():
            out = comparison + "avg"

    if wts is not None:
        if f"{out}wts" in estimands.keys():
            out = out + "wts"

    lab = {
        "difference": "{hi} - {lo}",
        "differenceavg": "mean({hi}) - mean({lo})",
        "differenceavgwts": "mean({hi}) - mean({lo})",
        "dydx": "dY/dX",
        "eyex": "eY/eX",
        "eydx": "eY/dX",
        "dyex": "dY/eX",
        "dydxavg": "mean(dY/dX)",
        "eyexavg": "mean(eY/eX)",
        "eydxavg": "mean(eY/dX)",
        "dyexavg": "mean(dY/eX)",
        "dydxavgwts": "mean(dY/dX)",
        "eyexavgwts": "mean(eY/eX)",
        "eydxavgwts": "mean(eY/dX)",
        "dyexavgwts": "mean(dY/eX)",
        "ratio": "{hi} / {lo}",
        "ratioavg": "mean({hi}) / mean({lo})",
        "ratioavgwts": "mean({hi}) / mean({lo})",
        "lnratio": "ln({hi} / {lo})",
        "lnratioavg": "ln(mean({hi}) / mean({lo}))",
        "lnratioavgwts": "ln(mean({hi}) / mean({lo}))",
        "lnor": "ln(odds({hi}) / odds({lo}))",
        "lnoravg": "ln(odds({hi}) / odds({lo}))",
        "lnoravgwts": "ln(odds({hi}) / odds({lo}))",
        "lift": "lift",
        "liftavg": "liftavg",
        "expdydx": "exp(dY/dX)",
    }

    return (out, lab[out])


HiLo = namedtuple("HiLo", ["variable", "hi", "lo", "lab", "pad", "comparison"])



def clean_global(k, n):
    if (
        not isinstance(k, list)
        and not isinstance(k, pl.Series)
        and not isinstance(k, np.ndarray)
    ):
        out = [k]
    if not isinstance(k, list) or len(k) == 1:
        out = pl.Series(np.repeat(k, n))
    else:
        out = pl.Series(k)
    return out


def get_one_variable_hi_lo(variable, value, newdata, comparison, eps, by, wts=None, modeldata=None):
    msg = "`value` must be a numeric, a list of length two, or 'sd'"
    vartype = get_variable_type(variable, newdata)

    def clean(k):
        return clean_global(k, newdata.shape[0])

    elasticities = [
        "eyexavg",
        "dyexavg",
        "eydxavg",
        "dydxavg",
        "eyex",
        "dyex",
        "eydx",
        "dydx",
    ]

    # default
    if value is None:
        # derivatives are not supported for character or boolean variables
        if vartype in ["character", "boolean"]:
            value = "reference"
            if comparison in ["eyexavg", "dyexavg", "eydxavg", "dydxavg"]:
                comparison = "differenceavg"
            elif comparison in ["eyex", "dyex", "eydx", "dydx"]:
                comparison = "difference"
        else:
            if comparison in elasticities:
                value = eps
            else:
                value = 1

    comparison, lab = sanitize_comparison(comparison, by, wts)

    if vartype == "boolean":
        hi = clean(True)
        lo = clean(False)
        lab = lab.format(hi="True", lo="False")
        out = HiLo(
            variable=variable, hi=hi, lo=lo, lab=lab, comparison=comparison, pad=None
        )
        return [out]

    if vartype == "binary":
        hi = clean(1)
        lo = clean(0)
        lab = lab.format(hi="1", lo="0")
        out = HiLo(
            variable=variable, hi=hi, lo=lo, lab=lab, comparison=comparison, pad=None
        )
        return [out]

    if vartype == "character":
        if isinstance(value, list) and len(value) == 2:
            hi = clean([value[1]])
            lo = clean([value[0]])
            lab = lab.format(hi=hi, lo=lo)
            out = HiLo(
                variable=variable,
                hi=hi,
                lo=lo,
                lab=lab,
                comparison=comparison,
                pad=None,
            )
            return [out]

        elif isinstance(value, str):
            out = get_categorical_combinations(
                variable=variable,
                uniqs=modeldata[variable].unique().sort(),
                newdata=newdata,
                combo=value,
                comparison=comparison,
            )
            return out

        else:
            raise ValueError(msg)

    if vartype == "numeric" and isinstance(value, str):
        if value == "sd":
            value = newdata[variable].std()
            hi = newdata[variable] + value / 2
            lo = newdata[variable] - value / 2
            lab = lab.format(hi="(x+sd/2)", lo="(x-sd/2)")
        elif value == "2sd":
            value = newdata[variable].std()
            hi = newdata[variable] + value
            lo = newdata[variable] - value
            lab = lab.format(hi="(x+sd)", lo="(x-sd)")
        elif value == "iqr":
            hi = np.percentile(newdata[variable], 75)
            lo = np.percentile(newdata[variable], 25)
            lab = lab.format(hi="Q3", lo="Q1")
        elif value == "minmax":
            hi = np.max(newdata[variable])
            lo = np.min(newdata[variable])
            lab = lab.format(hi="Max", lo="Min")
        else:
            raise ValueError(msg)

    elif isinstance(value, list):
        if len(value) != 2:
            raise ValueError(msg)
        hi = clean([value[1]])
        lo = clean([value[0]])
        lab = lab.format(hi=value[1], lo=value[0])

    elif isinstance(value, (int, float)):
        if comparison not in elasticities:
            lab = f"+{value}"
        hi = newdata[variable] + value / 2
        lo = newdata[variable] - value / 2

    else:
        raise ValueError(msg)

    if isinstance(value, list):
        lo = clean([value[0]])
        hi = clean([value[1]])
        lab = lab.format(value[1], value[0])
    else:
        lo = newdata[variable] - value / 2
        hi = newdata[variable] + value / 2
        if comparison not in elasticities:
            lab = f"+{value}"

    if len(lo) == 1:
        if comparison not in elasticities:
            lab = lab.format(hi=hi[0], lo=lo[0])
        lo = clean(np.repeat(lo, newdata.shape[0]))
        hi = clean(np.repeat(hi, newdata.shape[0]))

    out = [
        HiLo(variable=variable, lo=lo, hi=hi, lab=lab, pad=None, comparison=comparison)
    ]
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
        assert isinstance(
            variables, dict
        ), "`variables` must be None, a dict, string, or list of strings"
    good = [x for x in variables if x in newdata.columns]
    bad = [x for x in variables if x not in newdata.columns]
    if len(bad) > 0:
        bad = ", ".join(bad)
        warn(f"Variable(s) not in newdata: {bad}")
    if len(good) == 0:
        raise ValueError("There is no valid column name in `variables`.")
    return variables


def get_categorical_combinations(
    variable, uniqs, newdata, comparison, combo="reference"
):
    def clean(k):
        return clean_global(k, newdata.shape[0])

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
                    hi=clean([u]),
                    lo=clean([uniqs[0]]),
                    lab=f"{u} - {uniqs[0]}",
                    pad=uniqs,
                    comparison=comparison,
                )
                out.append(hl)
    elif combo == "revreference":
        last_element = uniqs[-1]
        for u in uniqs:
            if u != last_element:
                hl = HiLo(
                    variable=variable,
                    hi=clean([u]),
                    lo=clean([last_element]),
                    lab=f"{u} - {last_element}",
                    comparison=comparison,
                    pad=uniqs,
                )
                out.append(hl)
    elif combo == "sequential":
        for i in range(len(uniqs) - 1):
            hl = HiLo(
                variable=variable,
                hi=clean([uniqs[i + 1]]),
                lo=clean([uniqs[i]]),
                lab=f"{uniqs[i + 1]} - {uniqs[i]}",
                comparison=comparison,
                pad=uniqs,
            )
            out.append(hl)
    elif combo == "revsequential":
        for i in range(len(uniqs) - 1, 0, -1):
            hl = HiLo(
                variable=variable,
                hi=clean([uniqs[i - 1]]),
                lo=clean([uniqs[i]]),
                lab=f"{uniqs[i - 1]} - {uniqs[i]}",
                comparison=comparison,
                pad=uniqs,
            )
            out.append(hl)
    elif combo == "pairwise":
        for i in range(len(uniqs)):
            for j in range(i + 1, len(uniqs)):
                hl = HiLo(
                    variable=variable,
                    hi=clean([uniqs[j]]),
                    lo=clean([uniqs[i]]),
                    lab=f"{uniqs[j]} - {uniqs[i]}",
                    comparison=comparison,
                    pad=uniqs,
                )
                out.append(hl)
    elif combo == "revpairwise":
        for i in range(len(uniqs)):
            for j in range(i + 1, len(uniqs)):
                hl = HiLo(
                    variable=variable,
                    hi=clean([uniqs[i]]),
                    lo=clean([uniqs[j]]),
                    lab=f"{uniqs[i]} - {uniqs[j]}",
                    comparison=comparison,
                    pad=uniqs,
                )
                out.append(hl)
    else:
        raise ValueError(
            "The supported comparisons are: 'reference', 'revreference', 'sequential', 'revsequential', 'pairwise', and 'revpairwise'."
        )

    return out


def sanitize_variables(variables, model, newdata, comparison, eps, by, wts=None):
    out = []

    modeldata = get_modeldata(model)

    if variables is None:
        vlist = get_variables_names(variables, model, newdata)
        vlist.sort()
        for v in vlist:
            out.append(
                get_one_variable_hi_lo(v, None, newdata, comparison, eps, by, wts, modeldata=modeldata)
            )

    elif isinstance(variables, dict):
        for v in variables:
            if v not in newdata.columns:
                del variables[v]
                warn(f"Variable {v} is not in newdata.")
            else:
                out.append(
                    get_one_variable_hi_lo(
                        v, variables[v], newdata, comparison, eps, by, wts, modeldata=modeldata
                    )
                )

    elif isinstance(variables, str):
        if variables not in newdata.columns:
            raise ValueError(f"Variable {variables} is not in newdata.")
        out.append(
            get_one_variable_hi_lo(variables, None, newdata, comparison, eps, by, wts, modeldata=modeldata)
        )

    elif isinstance(variables, list):
        for v in variables:
            if v not in newdata.columns:
                warn(f"Variable {v} is not in newdata.")
            else:
                out.append(
                    get_one_variable_hi_lo(v, None, newdata, comparison, eps, by, wts, modeldata=modeldata)
                )

    # unnest list of list of HiLo
    out = [item for sublist in out for item in sublist]

    return out
