"""
JAX dispatch layer for marginaleffects.

Provides functions to attempt JAX-based computation of predictions/comparisons,
returning None if JAX cannot be used (triggering fallback to finite differences).
"""

from typing import Optional, Dict, Any
import numpy as np

# Mapping from comparison string to ComparisonType enum value
# Only includes non-averaging comparisons supported by the autodiff module
# Averaging comparisons (differenceavg, etc.) have complex aggregation logic
COMPARISON_TYPE_MAP = {
    "difference": 0,  # ComparisonType.DIFFERENCE
    "ratio": 1,  # ComparisonType.RATIO
    "lnratio": 2,  # ComparisonType.LNRATIO
    "lnor": 3,  # ComparisonType.LNOR
    "lift": 4,  # ComparisonType.LIFT
}


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


def try_jax_comparisons(
    model,
    hi_X: np.ndarray,
    lo_X: np.ndarray,
    vcov: Optional[np.ndarray],
    by,
    wts,
    hypothesis,
    comparison: str,
    cross: bool,
    nd=None,  # DataFrame with term/contrast info for by=True
) -> Optional[Dict[str, Any]]:
    """
    Attempt JAX-based comparisons with jacobian and standard errors.

    Returns None if JAX cannot be used. Otherwise returns dict with:
    - estimate: np.ndarray of comparison estimates
    - jacobian: np.ndarray
    - std_error: np.ndarray of standard errors
    - For by=True: also includes 'terms' and 'contrasts' arrays

    Conditions that prevent JAX usage (returns None):
    - Global autodiff setting is disabled
    - Model doesn't have get_autodiff_config() method
    - Model's autodiff config is None (unsupported model type)
    - wts argument is not None (weights not supported)
    - hypothesis is not None (hypothesis testing not supported in fast path)
    - by is a complex aggregation (only False and ['group'] supported)
    - vcov is None (no SEs needed anyway)
    - comparison is not a supported string type
    - cross is True (cross comparisons not supported)
    """
    from .settings import is_autodiff_enabled

    # Check global setting first
    if not is_autodiff_enabled():
        return None

    # Check basic requirements
    if vcov is None:
        return None

    if wts is not None:
        return None

    if hypothesis is not None:
        return None

    if cross:
        return None

    # Only support simple comparison strings
    if not isinstance(comparison, str):
        return None

    comparison_lower = comparison.lower()
    if comparison_lower not in COMPARISON_TYPE_MAP:
        return None

    comparison_type = COMPARISON_TYPE_MAP[comparison_lower]

    # Check by parameter
    is_by_false = by is False
    is_by_true = isinstance(by, list) and by == ["group"]
    if not is_by_false and not is_by_true:
        return None

    # For by=True, we need the nd DataFrame to get term groupings
    if is_by_true and nd is None:
        return None

    # Check model compatibility
    if not hasattr(model, "get_autodiff_config"):
        return None

    config = model.get_autodiff_config()
    if config is None:
        return None

    try:
        beta = model.get_coef()
        X_hi = np.asarray(hi_X)
        X_lo = np.asarray(lo_X)
        V = np.asarray(vcov)

        # Select appropriate autodiff module based on model type
        if config["model_type"] == "linear":
            from .autodiff import linear as ad_module
        elif config["model_type"] == "glm":
            from .autodiff import glm as ad_module
        else:
            return None

        if is_by_false:
            # Row-level comparisons
            if config["model_type"] == "linear":
                result = ad_module.comparisons.comparisons(
                    beta=beta,
                    X_hi=X_hi,
                    X_lo=X_lo,
                    vcov=V,
                    comparison_type=comparison_type,
                )
            else:
                result = ad_module.comparisons.comparisons(
                    beta=beta,
                    X_hi=X_hi,
                    X_lo=X_lo,
                    vcov=V,
                    comparison_type=comparison_type,
                    family_type=config["family_type"],
                    link_type=config["link_type"],
                )

            return {
                "estimate": result["estimate"],
                "jacobian": result["jacobian"],
                "std_error": result["std_error"],
            }

        elif is_by_true:
            # Per-term aggregation
            terms = nd["term"].to_numpy()
            contrasts = nd["contrast"].to_numpy()

            # Get unique term-contrast combinations (preserving order)
            seen = {}
            unique_pairs = []
            for i, (t, c) in enumerate(zip(terms, contrasts)):
                key = (t, c)
                if key not in seen:
                    seen[key] = len(unique_pairs)
                    unique_pairs.append(key)

            # Compute averaged comparison for each term-contrast pair
            estimates = []
            std_errors = []
            jacobians = []

            for term_val, contrast_val in unique_pairs:
                # Get indices for this term-contrast pair
                mask = (terms == term_val) & (contrasts == contrast_val)
                idx = np.where(mask)[0]

                # Slice matrices for this group
                X_hi_group = X_hi[idx]
                X_lo_group = X_lo[idx]

                # Compute averaged comparison for this group
                if config["model_type"] == "linear":
                    result = ad_module.comparisons.comparisons_byT(
                        beta=beta,
                        X_hi=X_hi_group,
                        X_lo=X_lo_group,
                        vcov=V,
                        comparison_type=comparison_type,
                    )
                else:
                    result = ad_module.comparisons.comparisons_byT(
                        beta=beta,
                        X_hi=X_hi_group,
                        X_lo=X_lo_group,
                        vcov=V,
                        comparison_type=comparison_type,
                        family_type=config["family_type"],
                        link_type=config["link_type"],
                    )

                estimates.append(float(result["estimate"]))
                std_errors.append(float(result["std_error"]))
                jacobians.append(result["jacobian"])

            return {
                "estimate": np.array(estimates),
                "jacobian": np.vstack(jacobians) if jacobians else np.array([]),
                "std_error": np.array(std_errors),
                "terms": [p[0] for p in unique_pairs],
                "contrasts": [p[1] for p in unique_pairs],
            }

    except Exception:
        # Any error -> fall back to finite differences
        return None
