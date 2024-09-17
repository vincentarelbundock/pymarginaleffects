import os
import re
from matplotlib.testing.compare import compare_images
from plotnine import ggsave
import warnings


from marginaleffects import *


def compare_r_to_py(r_obj, py_obj, tolr=1e-3, tola=1e-3, msg=""):
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


def assert_image(fig, label, folder, tolerance=5):
    known_path = f"./tests/images/{folder}/"
    unknown_path = f"./tests/images/.tmp_{folder}/"
    if os.path.isdir(unknown_path):
        for root, dirs, files in os.walk(unknown_path):
            for fname in files:
                os.remove(os.path.join(root, fname))
        os.rmdir(unknown_path)
    os.mkdir(unknown_path)
    unknown = f"{unknown_path}{label}.png"
    known = f"{known_path}{label}.png"
    if not os.path.exists(known):
        ggsave(fig, filename=known, verbose=False, height=5, width=10, dpi=100)
        warnings.warn(f"File {known} does not exist. Creating it now.")
        return None
    ggsave(fig, filename=unknown, verbose=False, height=5, width=10, dpi=100)
    out = compare_images(known, unknown, tol=tolerance)
    # os.remove(unknown)
    return out
