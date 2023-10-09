import os
import polars as pl
import statsmodels.formula.api as smf
import pytest
from matplotlib.testing.compare import compare_images
from marginaleffects import *
from marginaleffects.plot_comparisons import *
from .utilities import *


df = pl.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/palmerpenguins/penguins.csv", null_values = "NA") \
    .drop_nulls()
mod = smf.ols("body_mass_g ~ flipper_length_mm * species * bill_length_mm + island", df).fit()

@pytest.mark.skip(reason="statsmodels vcov is weird")
def test_plot_comparisons():

    tolerance = 0.05

    baseline_path = "./tests/images/plot_comparisons/"
    
    result_path = "./tests/images/.tmp_plot_comparisons/"
    if os.path.isdir(result_path):
        for root, dirs, files in os.walk(result_path):
            for fname in files:
                os.remove(os.path.join(root, fname))
        os.rmdir(result_path)
    os.mkdir(result_path)

    fig = plot_comparisons(mod, variables='species', by='island')
    fig.savefig(result_path + "Figure_1.png")
    assert compare_images(baseline_path + "Figure_1.png", result_path + "Figure_1.png", tolerance) is None
    os.remove(result_path + "Figure_1.png")

    fig = plot_comparisons(mod, variables='bill_length_mm', newdata=datagrid(mod, bill_length_mm=[37,39]), by='island')
    fig.savefig(result_path + "Figure_2.png")
    assert compare_images(baseline_path + "Figure_2.png", result_path + "Figure_2.png", tolerance) is None
    os.remove(result_path + "Figure_2.png")

    fig = plot_comparisons(mod, variables='bill_length_mm', condition=['flipper_length_mm', 'species'])
    fig.savefig(result_path + "Figure_3.png")
    assert compare_images(baseline_path + "Figure_3.png", result_path + "Figure_3.png", tolerance) is None
    os.remove(result_path + "Figure_3.png")

    fig = plot_comparisons(mod, variables='species', condition='bill_length_mm')
    fig.savefig(result_path + "Figure_4.png")
    assert compare_images(baseline_path + "Figure_4.png", result_path + "Figure_4.png", tolerance) is None
    os.remove(result_path + "Figure_4.png")

    fig = plot_comparisons(mod, variables='island', condition='bill_length_mm')
    fig.savefig(result_path + "Figure_5.png")
    assert compare_images(baseline_path + "Figure_5.png", result_path + "Figure_5.png", tolerance) is None
    os.remove(result_path + "Figure_5.png")

    fig = plot_comparisons(mod, variables='species', condition=['bill_length_mm', 'species', 'island'])
    fig.savefig(result_path + "Figure_6.png")
    assert compare_images(baseline_path + "Figure_6.png", result_path + "Figure_6.png", tolerance) is None
    os.remove(result_path + "Figure_6.png")

    os.rmdir(result_path)

    return