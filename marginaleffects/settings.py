"""
Global settings for marginaleffects.
"""

import os
from typing import Optional

# Internal state
_settings = {
    "autodiff": None,  # None = auto-detect, True = force on, False = force off
}


def set_autodiff(enabled: Optional[bool]) -> None:
    """
    Enable or disable JAX-based automatic differentiation.

    Parameters
    ----------
    enabled : bool or None
        - True: Force JAX usage (error if JAX not installed)
        - False: Disable JAX, always use finite differences
        - None: Auto-detect (use JAX if available and model is compatible)

    Examples
    --------
    >>> import marginaleffects as me
    >>> me.set_autodiff(False)  # Disable JAX
    >>> me.set_autodiff(True)   # Force JAX (raises if not installed)
    >>> me.set_autodiff(None)   # Auto-detect (default)
    """
    if enabled is not None and not isinstance(enabled, bool):
        raise TypeError("`enabled` must be True, False, or None")
    _settings["autodiff"] = enabled


def get_autodiff() -> Optional[bool]:
    """
    Get the current autodiff setting.

    Returns
    -------
    bool or None
        Current setting. Also checks MARGINALEFFECTS_AUTODIFF env var
        if no programmatic setting has been made.
    """
    # Programmatic setting takes precedence
    if _settings["autodiff"] is not None:
        return _settings["autodiff"]

    # Check environment variable
    env_val = os.environ.get("MARGINALEFFECTS_AUTODIFF", "").lower()
    if env_val in ("0", "false", "no", "off"):
        return False
    if env_val in ("1", "true", "yes", "on"):
        return True

    # Default: auto-detect (None)
    return None


def is_autodiff_enabled() -> bool:
    """
    Check if JAX autodiff should be attempted.

    Returns False if:
    - User explicitly disabled via set_autodiff(False) or env var
    - JAX is not installed and user didn't force it on

    Returns True if:
    - User explicitly enabled via set_autodiff(True)
    - Auto-detect mode and JAX is available
    """
    setting = get_autodiff()

    if setting is False:
        return False

    if setting is True:
        # User wants JAX - check if available
        try:
            from .autodiff import _JAX_AVAILABLE

            if not _JAX_AVAILABLE:
                raise ImportError(
                    "JAX autodiff was explicitly enabled but JAX is not installed. "
                    "Install with: pip install marginaleffects[autodiff]"
                )
            return True
        except ImportError as e:
            raise ImportError(str(e)) from None

    # Auto-detect mode (setting is None)
    try:
        from .autodiff import _JAX_AVAILABLE

        return _JAX_AVAILABLE
    except ImportError:
        return False
