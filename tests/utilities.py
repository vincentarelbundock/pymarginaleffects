import re

import numpy as np
import polars as pl

from marginaleffects import *


def compare_r_to_py(r_obj, py_obj, tolr = 1e-3, tola = 1e-3, msg = ""):
    cols = ["term", "contrast", "rowid"]
    cols = [x for x in cols if x in r_obj.columns and x in py_obj.columns]
    r_obj = r_obj.sort(cols)
    py_obj = py_obj.sort(cols)
    # dont' compare other statistics because degrees of freedom don't match
    # for col_py in ["estimate", "std_error"]:
    for col_py in ["estimate"]:
        col_r = re.sub("_", ".", col_py) 
        if col_py in py_obj.columns and col_r in r_obj.columns:
            a = r_obj[col_r]
            b = py_obj[col_py]
            gap_rel = ((a - b) / a).abs().max()
            gap_abs = (a - b).abs().max()
            flag = gap_rel <= tolr or gap_abs <= tola
            assert flag, f"{msg} trel: {gap_rel}. tabs: {gap_abs}"
