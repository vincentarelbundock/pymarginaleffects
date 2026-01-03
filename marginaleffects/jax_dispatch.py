"""
JAX dispatch layer for marginaleffects.

Provides functions to attempt JAX-based computation of predictions/comparisons,
returning None if JAX cannot be used (triggering fallback to finite differences).
"""

from typing import Optional, Dict, Any
import numpy as np


def try_jax_predictions(
    model,
    exog: np.ndarray,
    vcov: Optional[np.ndarray],
    by,
    wts,
    hypothesis,
) -> Optional[Dict[str, Any]]:
    """
    Attempt JAX-based prediction with jacobian and standard errors.

    Returns None if JAX cannot be used. Otherwise returns dict with:
    - estimate: np.ndarray of predictions
    - jacobian: np.ndarray (n_obs x n_coefs)
    - std_error: np.ndarray of standard errors

    Conditions that prevent JAX usage (returns None):
    - Global autodiff setting is disabled
    - Model doesn't have get_autodiff_config() method
    - Model's autodiff config is None (unsupported model type)
    - wts argument is not None (weights not supported)
    - hypothesis is not None (hypothesis testing not supported in fast path)
    - by is a list/complex aggregation (only False and True supported initially)
    - vcov is None (no SEs needed anyway)
    """
    from .settings import is_autodiff_enabled

    # Check global setting first
    if not is_autodiff_enabled():
        return None

    # Check basic requirements
    if vcov is None:
        return None  # No point using JAX if we don't need SEs

    if wts is not None:
        return None  # Weights not supported

    if hypothesis is not None:
        return None  # Hypothesis testing uses different path

    # Only support by=False or by=True (sanitized to ['group']) for now
    # After sanitize_by(): False stays False, True becomes ['group']
    is_by_false = by is False
    is_by_true = isinstance(by, list) and by == ["group"]
    if not is_by_false and not is_by_true:
        return None

    # Check model compatibility
    if not hasattr(model, "get_autodiff_config"):
        return None

    config = model.get_autodiff_config()
    if config is None:
        return None

    try:
        beta = model.get_coef()
        X = np.asarray(exog)
        V = np.asarray(vcov)

        # Select appropriate autodiff module based on model type
        if config["model_type"] == "linear":
            from .autodiff import linear as ad_module

            if is_by_false:
                # Row-level predictions
                result = ad_module.predictions.predictions(
                    beta=beta,
                    X=X,
                    vcov=V,
                )
            elif is_by_true:
                # Average prediction (single value)
                result = ad_module.predictions.predictions_byT(
                    beta=beta,
                    X=X,
                    vcov=V,
                )

        elif config["model_type"] == "glm":
            from .autodiff import glm as ad_module

            if is_by_false:
                # Row-level predictions
                result = ad_module.predictions.predictions(
                    beta=beta,
                    X=X,
                    vcov=V,
                    family_type=config["family_type"],
                    link_type=config["link_type"],
                )
            elif is_by_true:
                # Average prediction (single value)
                result = ad_module.predictions.predictions_byT(
                    beta=beta,
                    X=X,
                    vcov=V,
                    family_type=config["family_type"],
                    link_type=config["link_type"],
                )
        else:
            return None

        return {
            "estimate": result["estimate"],
            "jacobian": result["jacobian"],
            "std_error": result["std_error"],
        }

    except Exception:
        # Any error -> fall back to finite differences
        return None
