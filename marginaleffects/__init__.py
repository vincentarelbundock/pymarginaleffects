from .comparisons import avg_comparisons, comparisons
from .datagrid import datagrid
from .hypotheses import hypotheses
from .model_linearmodels import fit_linearmodels
from .model_sklearn import fit_sklearn
from .model_statsmodels import fit_statsmodels
from .plot_comparisons import plot_comparisons
from .plot_predictions import plot_predictions
from .plot_slopes import plot_slopes
from .predictions import avg_predictions, predictions
from .slopes import avg_slopes, slopes
from .utils import get_dataset
from .result import MarginaleffectsResult

# Conditionally import autodiff module if JAX is available
try:
    from . import autodiff

    _AUTODIFF_AVAILABLE = True
except ImportError:
    _AUTODIFF_AVAILABLE = False
    autodiff = None

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
]

if _AUTODIFF_AVAILABLE:
    __all__.append("autodiff")
