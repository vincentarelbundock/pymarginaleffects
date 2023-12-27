from collections import namedtuple
from warnings import warn

import numpy as np
import polars as pl

from .datagrid import datagrid
from .estimands import estimands
from .utils import get_variable_type


def sanitize_vcov(vcov, model):
    V = model.get_vcov(vcov)
    if V is not None:
        assert isinstance(V, np.ndarray), "vcov must be True or a square NumPy array"
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
        raise ValueError(
            "The `by` argument must be True, False, a string, or a list of strings."
        )
    return by


def sanitize_newdata(model, newdata, wts, by=[]):
    modeldata = model.modeldata

    if newdata is None:
        out = modeldata

    elif isinstance(newdata, str) and newdata == "mean":
        out = datagrid(newdata=modeldata)

    elif isinstance(newdata, str) and newdata == "median":
        out = datagrid(newdata=modeldata, FUN_numeric=lambda x: x.median())

    else:
        try:
            import pandas as pd

            if isinstance(newdata, pd.DataFrame):
                out = pl.from_pandas(newdata)
            else:
                out = newdata
        except ImportError:
            out = newdata

    datagrid_explicit = None
    if isinstance(out, pl.DataFrame) and hasattr(out, "datagrid_explicit"):
        datagrid_explicit = out.datagrid_explicit

    if isinstance(by, list) and len(by) > 0:
        by = [x for x in by if x in out.columns]
        if len(by) > 0:
            out = out.sort(by)

    out = out.with_columns(pl.Series(range(out.height), dtype=pl.Int32).alias("rowid"))

    if wts is not None:
        if (isinstance(wts, str) is False) or (wts not in out.columns):
            raise ValueError(f"`newdata` does not have a column named '{wts}'.")

    xnames = model.get_variables_names(variables=None, newdata=modeldata)
    ynames = model.response_name
    if isinstance(ynames, str):
        ynames = [ynames]
    cols = [x for x in xnames + ynames if x in out.columns]
    out = out.drop_nulls(subset=cols)

    if any([isinstance(out[x], pl.Categorical) for x in out.columns]):
        raise ValueError("Categorical type columns are not supported in `newdata`.")

    if datagrid_explicit is not None:
        out.datagrid_explicit = datagrid_explicit

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

    assert (
        out in lab.keys()
    ), f"`comparison` must be one of: {', '.join(list(lab.keys()))}."

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


def get_one_variable_hi_lo(
    variable, value, newdata, comparison, eps, by, wts=None, modeldata=None
):
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
                lab=lab,
                combo=value,
                comparison=comparison,
            )
            return out

        else:
            raise ValueError(msg)

    if vartype == "numeric":
        if isinstance(value, str):
            if value == "sd":
                value = modeldata[variable].std()
                hi = newdata[variable] + value / 2
                lo = newdata[variable] - value / 2
                lab = lab.format(hi="(x+sd/2)", lo="(x-sd/2)")
            elif value == "2sd":
                value = modeldata[variable].std()
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

        out = [
            HiLo(
                variable=variable,
                lo=lo,
                hi=hi,
                lab=lab,
                pad=None,
                comparison=comparison,
            )
        ]
        return out

    raise ValueError(msg)


def get_categorical_combinations(
    variable, uniqs, newdata, comparison, lab, combo="reference"
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
                    lab=lab.format(hi=u, lo=uniqs[0]),
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
                    lab=lab.format(hi=u, lo=last_element),
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
                lab=lab.format(hi=uniqs[i + 1], lo=uniqs[i]),
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
                lab=lab.format(hi=uniqs[i - 1], lo=uniqs[i]),
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
                    lab=lab.format(hi=uniqs[j], lo=uniqs[i]),
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
                    lab=lab.format(hi=uniqs[i], lo=uniqs[j]),
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

    modeldata = model.modeldata

    if variables is None:
        vlist = model.get_variables_names(variables, modeldata)
        vlist.sort()
        for v in vlist:
            out.append(
                get_one_variable_hi_lo(
                    v, None, newdata, comparison, eps, by, wts, modeldata=modeldata
                )
            )

    elif isinstance(variables, dict):
        for v in variables:
            if v not in newdata.columns:
                del variables[v]
                warn(f"Variable {v} is not in newdata.")
            else:
                out.append(
                    get_one_variable_hi_lo(
                        v,
                        variables[v],
                        newdata,
                        comparison,
                        eps,
                        by,
                        wts,
                        modeldata=modeldata,
                    )
                )

    elif isinstance(variables, str):
        if variables not in newdata.columns:
            raise ValueError(f"Variable {variables} is not in newdata.")
        out.append(
            get_one_variable_hi_lo(
                variables, None, newdata, comparison, eps, by, wts, modeldata=modeldata
            )
        )

    elif isinstance(variables, list):
        for v in variables:
            if v not in newdata.columns:
                warn(f"Variable {v} is not in newdata.")
            else:
                out.append(
                    get_one_variable_hi_lo(
                        v, None, newdata, comparison, eps, by, wts, modeldata=modeldata
                    )
                )

    # unnest list of list of HiLo
    out = [item for sublist in out for item in sublist]

    return out


def sanitize_hypothesis_null(hypothesis):
    if isinstance(hypothesis, (int, float)):
        hypothesis_null = hypothesis
    else:
        hypothesis_null = 0
    return hypothesis_null
