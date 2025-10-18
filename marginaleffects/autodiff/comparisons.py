"""Comparison types and functions using enum-based approach for JAX compatibility."""

import jax.numpy as jnp
from jax import lax
from enum import IntEnum


class ComparisonType(IntEnum):
    """Comparison types for marginal effects."""

    DIFFERENCE = 0
    RATIO = 1
    LNRATIO = 2
    LNOR = 3
    LIFT = 4
    DIFFERENCEAVG = 5


def _compute_comparison_vector(
    comparison_type: int, pred_hi: jnp.ndarray, pred_lo: jnp.ndarray
) -> jnp.ndarray:
    """Apply comparison function element-wise (returns N-length array)."""
    return lax.switch(
        comparison_type,
        [
            lambda hi, lo: hi - lo,  # difference
            lambda hi, lo: hi / lo,  # ratio
            lambda hi, lo: jnp.log(hi / lo),  # lnratio
            lambda hi, lo: jnp.log((hi / (1 - hi)) / (lo / (1 - lo))),  # lnor
            lambda hi, lo: (hi - lo) / lo,  # lift
        ],
        pred_hi,
        pred_lo,
    )


def _compute_comparison_scalar(
    comparison_type: int, pred_hi: jnp.ndarray, pred_lo: jnp.ndarray
) -> jnp.ndarray:
    """Apply comparison function with aggregation (returns scalar)."""
    return lax.switch(
        comparison_type,
        [
            lambda hi, lo: jnp.mean(hi) - jnp.mean(lo),  # differenceavg
        ],
        pred_hi,
        pred_lo,
    )


def _compute_comparison(
    comparison_type: int, pred_hi: jnp.ndarray, pred_lo: jnp.ndarray
) -> jnp.ndarray:
    """Apply comparison function element-wise (returns N-length array).

    Note: DIFFERENCEAVG should use _compute_comparison_scalar directly.
    """
    return _compute_comparison_vector(comparison_type, pred_hi, pred_lo)
