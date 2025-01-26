"""Create color gradients."""
from __future__ import annotations

from typing import Literal as _Literal, Sequence as _Sequence, TYPE_CHECKING as _TYPE_CHECKING


import numpy as _np

from pycolorit.color import RGBColor as _RGBColor
from pycolorit import color as _color


def interpolate(
    color_start: _RGBColor,
    color_end: _RGBColor,
    count: int = 3,
    system: _Literal["rgb", "hsl", "hsv", "hwb"] = "rgb",
    multipliers: _Sequence[float] | None = None,
    hue_wrap: bool = False,
    return_array: bool = False,
) -> list[_RGBColor] | _np.ndarray:
    """Create a color gradient between two RGB colors.

    Parameters
    ----------
    color_start : RGBColor
        Initial RGB color in the gradient.
    color_end : RGBColor
        Final RGB color in the gradient.
    count : positive integer, default: 3
        Total number of returned colors. The initial and final colors,
        which are always part of the gradient, are included, i.e.
        for example `count=1` returns `[color_start]`, `count=2` returns
        `[color_start, color_end]`, and `count=3` returns
        `[color_start, intermediate_color_1, color_end]`.
    system : {'rgb', 'hsl', 'hsv', 'hwb'}, default: 'rgb'
        Coordinate system to use for interpolation.
    multipliers : sequence of 3 or 4 numbers, optional
        Multipliers for each component of the coordinate system;
        either 3 numbers for the main components (e.g. RGB, HSL, HSV, HWB),
        or 4 numbers to also include the alpha channel.
    hue_wrap : bool, default: False
        Only applicable to HSL, HSV, and HWB systems:
        If True, the hue values of both start and end colors are wrapped before interpolation.
    return_array : bool, default: False
        If True, return the colors as a numpy array of color components instead of a list of RGBColor objects.
    Returns
    -------
    colors : list of RGBColor, or numpy.ndarray
        Gradient colors starting with `color_start`.
    """
    explicit_alpha = multipliers and len(multipliers) == 4
    if not isinstance(color_start, _RGBColor):
        color_start = _color.css(str(color_start))
    if not isinstance(color_end, _RGBColor):
        color_end = _color.css(str(color_end))
    c1 = color_start.array(
        system=system,
        hue_wrap=hue_wrap,
        explicit_alpha=explicit_alpha,
    )
    c2 = color_end.array(
        system=system,
        hue_wrap=hue_wrap,
        explicit_alpha=explicit_alpha
    )
    step = (c2 - c1) / (count - 1)
    if multipliers:
        step *= _np.array(multipliers)
    gradient = (_np.arange(count)[..., _np.newaxis] * step) + c1[_np.newaxis, ...]
    if return_array:
        return gradient
    return [_RGBColor(system=system, values=color) for color in gradient]
