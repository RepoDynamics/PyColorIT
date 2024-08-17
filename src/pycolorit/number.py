"""Utility functions for working with numbers."""

import numpy as _np


def clean_float(num: float | int | _np.number) -> float | int | _np.number:
    """Convert a float to an integer if it is an integer."""
    return int(num) if num.is_integer() else num
