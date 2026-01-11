from .comparisons import avg_comparisons, comparisons
from .datagrid import datagrid
from .hypotheses import hypotheses
from .linearmodels import fit_linearmodels
from .sklearn import fit_sklearn
from .statsmodels import fit_statsmodels
from .plot import plot_comparisons, plot_predictions, plot_slopes
from .predictions import avg_predictions, predictions
from .slopes import avg_slopes, slopes
from .utils import get_dataset
from .result import MarginaleffectsResult
from .settings import autodiff, set_autodiff, get_autodiff

# Conditionally import autodiff module if JAX is available
try:
    from . import autodiff as autodiff_module

    _AUTODIFF_MODULE_AVAILABLE = True
except ImportError:
    _AUTODIFF_MODULE_AVAILABLE = False
    autodiff_module = None

__all__ = [
    "avg_comparisons",
    "comparisons",
    "datagrid",
    "hypotheses",
    "plot_comparisons",
    "plot_predictions",
    "plot_slopes",
    "avg_predictions",
    "predictions",
    "avg_slopes",
    "slopes",
    "fit_statsmodels",
    "fit_sklearn",
    "fit_linearmodels",
    "get_dataset",
    "MarginaleffectsResult",
    "autodiff",
    "set_autodiff",
    "get_autodiff",
]

if _AUTODIFF_MODULE_AVAILABLE:
    __all__.append("autodiff_module")
