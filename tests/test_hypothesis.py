import numpy as np
import statsmodels.formula.api as smf
from marginaleffects import *

mtcars = get_dataset("mtcars", "datasets")
mod = smf.ols("mpg ~ hp + cyl", data=mtcars).fit()
p = predictions(mod, by="cyl")["estimate"]


def test_predictions_reference():
    q = predictions(mod, by="cyl", hypothesis="difference~reference")["estimate"]
    assert np.isclose(p[1] - p[0], q[0])
    assert np.isclose(p[2] - p[0], q[1])
    q = predictions(mod, by="cyl", hypothesis="ratio~reference")["estimate"]
    assert np.isclose(p[1] / p[0], q[0])
    assert np.isclose(p[2] / p[0], q[1])
    q = predictions(mod, by="cyl", hypothesis="difference~revreference")["estimate"]
    assert np.isclose(p[0] - p[1], q[0])
    assert np.isclose(p[0] - p[2], q[1])
    q = predictions(mod, by="cyl", hypothesis="ratio~revreference")["estimate"]
    assert np.isclose(p[0] / p[1], q[0])
    assert np.isclose(p[0] / p[2], q[1])


def test_predictions_sequential():
    q = predictions(mod, by="cyl", hypothesis="difference~sequential")["estimate"]
    assert np.isclose(p[1] - p[0], q[0])
    assert np.isclose(p[2] - p[1], q[1])
    q = predictions(mod, by="cyl", hypothesis="ratio~sequential")["estimate"]
    assert np.isclose(p[1] / p[0], q[0])
    assert np.isclose(p[2] / p[1], q[1])
    q = predictions(mod, by="cyl", hypothesis="difference~sequential")["estimate"]
    assert np.isclose(p[1] - p[0], q[0])
    assert np.isclose(p[2] - p[1], q[1])
    q = predictions(mod, by="cyl", hypothesis="ratio~sequential")["estimate"]
    assert np.isclose(p[1] / p[0], q[0])
    assert np.isclose(p[2] / p[1], q[1])


def test_predictions_pairwise():
    p = predictions(mod, by="cyl")["estimate"]
    q = predictions(mod, by="cyl", hypothesis="ratio~pairwise")["estimate"]
    assert np.isclose(p[1] / p[0], q[0])
    assert np.isclose(p[2] / p[0], q[1])
    assert np.isclose(p[2] / p[1], q[2])
    p = predictions(mod, by="cyl")["estimate"]
    q = predictions(mod, by="cyl", hypothesis="difference~pairwise")["estimate"]
    assert np.isclose(p[1] - p[0], q[0])
    assert np.isclose(p[2] - p[0], q[1])
    assert np.isclose(p[2] - p[1], q[2])
    p = predictions(mod, by="cyl")["estimate"]
    q = predictions(mod, by="cyl", hypothesis="ratio~revpairwise")["estimate"]
    assert np.isclose(p[0] / p[1], q[0])
    assert np.isclose(p[0] / p[2], q[1])
    assert np.isclose(p[1] / p[2], q[2])
    p = predictions(mod, by="cyl")["estimate"]
    q = predictions(mod, by="cyl", hypothesis="difference~revpairwise")["estimate"]
    assert np.isclose(p[0] - p[1], q[0])
    assert np.isclose(p[0] - p[2], q[1])
    assert np.isclose(p[1] - p[2], q[2])
