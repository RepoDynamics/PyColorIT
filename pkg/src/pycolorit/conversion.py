"""Conversion functions for color systems and hue angles.

References
----------
- [Python `colorsys` module](https://docs.python.org/3.12/library/colorsys.html)
- [Wikipedia - HSL and HSV](https://en.wikipedia.org/wiki/HSL_and_HSV)
- [StackOverflow - RGB to HSL conversion](https://stackoverflow.com/questions/39118528/rgb-to-hsl-conversion)
- [Wikipedia - HWB color model](https://en.wikipedia.org/wiki/HWB_color_model)
"""

from typing import Literal as _Literal, Sequence as _Sequence

import numpy as _np

from pycolorit import number as _number


class RGBConverter:
    """Convert RGB colors to other color systems in the RGB model."""

    def __init__(self, rgb: _Sequence[float | int] | _np.ndarray):
        """
        Parameters
        ----------
        rgb : array of numbers
            An array of RGB (shape: [..., 3]) or RGBA (shape: [..., 4]) values.
        """
        self._rgb = _np.asarray(rgb)
        color_dim = self._rgb.shape[-1]
        if color_dim not in (3, 4):
            raise ValueError("RGB values must have 3 or 4 components.")
        self._alpha = self._rgb[..., 3] if color_dim == 4 else None
        self._r, self._g, self._b = self._rgb[..., 0], self._rgb[..., 1], self._rgb[..., 2]
        self._max = None
        self._min = None
        self._minmax_sum = None
        self._chroma = None
        self._hue = None
        return

    def convert(self, target: _Literal["hsl", "hsv", "hwb"], decimals: int = 5) -> _np.ndarray:
        """Convert RGB to HSL, HSV, or HWB.

        Parameters
        ----------
        target : {'hsl', 'hsv', 'hwb'}
            Target color system to convert the RGB values to.
        decimals : int, default: 5
            Number of decimal places to round the target values to.

        Returns
        -------
        target_colors : numpy.ndarray
            Array of target color system values with the same shape as the input RGB array.
        """
        if target == "hsl":
            return self.to_hsl(decimals)
        if target == "hsv":
            return self.to_hsv(decimals)
        if target == "hwb":
            return self.to_hwb(decimals)
        raise ValueError(f"Conversion to {target.upper()} is not supported.")

    def to_hsl(self, decimals: int = 5) -> _np.ndarray:
        """Convert RGB to HSL.

        Parameters
        ----------
        decimals : int, default: 5
            Number of decimal places to round the HSL values to.

        Returns
        -------
        hsl_colors : numpy.ndarray
            Array of HSL values with the same shape as the input RGB array.
        """
        components = [self.hue, self.hsl_saturation, self.hsl_lightness]
        if self._alpha:
            components.append(self._alpha)
        return _np.round(_np.stack(components, axis=-1), decimals=decimals)

    def to_hsv(self, decimals: int = 5) -> _np.ndarray:
        """Convert RGB to HSV.

        Parameters
        ----------
        decimals : int, default: 5
            Number of decimal places to round the HSV values to.

        Returns
        -------
        hsv_colors : numpy.ndarray
            Array of HSV values with the same shape as the input RGB array.
        """
        components = [self.hue, self.hsv_saturation, self.hsv_value]
        if self._alpha:
            components.append(self._alpha)
        return _np.round(_np.stack(components, axis=-1), decimals=decimals)

    def to_hwb(self, decimals: int = 5) -> _np.ndarray:
        """Convert RGB to HWB.

        Parameters
        ----------
        decimals : int, default: 5
            Number of decimal places to round the HWB values to.

        Returns
        -------
        hwb_colors : numpy.ndarray
            Array of HWB values with the same shape as the input RGB array.
        """
        components = [self.hue, self.min, 1 - self.max]
        if self._alpha:
            components.append(self._alpha)
        return _np.round(_np.stack(components, axis=-1), decimals=decimals)

    @property
    def hue(self) -> _np.array:
        """HSL/HSV/HWB H (hue) values for each color."""
        if self._hue:
            return self._hue
        with _np.errstate(divide='ignore', invalid='ignore'):
            hue_raw = _np.where(
                self.chroma == 0,
                0,
                _np.where(
                    self.max == self._r,
                    (self._g - self._b) / self.chroma,
                    _np.where(
                        self.max == self._g,
                        2 + ((self._b - self._r) / self._chroma),
                        4 + ((self._r - self._g) / self.chroma)
                    ),
                ),
            )
        self._hue = _np.mod(hue_raw / 6, 1)
        return self._hue

    @property
    def hsl_saturation(self):
        """HSL S (saturation) values for each color."""
        # Suppress division by zero warnings: for some values, the denominator will be zero,
        # but those are not used (i.e., filtered by np.where).
        with _np.errstate(divide='ignore', invalid='ignore'):
            saturation = _np.where(
                self.chroma == 0,
                0,
                _np.where(
                    self.hsl_lightness > 0.5,
                    self.chroma / (2 - self.minmax_sum),
                    self.chroma / self.minmax_sum,
                )
            )
        return saturation

    @property
    def hsl_lightness(self):
        """HSL L (lightness) values for each color."""
        return self.minmax_sum / 2

    @property
    def hsv_saturation(self):
        """HSV S (saturation) values for each color."""
        with _np.errstate(divide='ignore', invalid='ignore'):
            saturation = _np.where(
                self.max == 0,
                0,
                self.chroma / self.max,
            )
        return saturation

    @property
    def hsv_value(self):
        """HSV V (value) values for each color."""
        return self.max

    @property
    def chroma(self):
        """Get the chroma value for each color,
        i.e., the difference between max. and min. RGB values.
        """
        if self._chroma is None:
            # Round to avoid floating point errors
            # see: https://github.com/python/cpython/pull/106530
            self._chroma = _np.round(self.max - self.min, decimals=15)
        return self._chroma

    @property
    def minmax_sum(self):
        """Get the sum of the maximum and minimum RGB values for each color."""
        if self._minmax_sum is None:
            # Round to avoid floating point errors
            # see: https://github.com/python/cpython/pull/106530
            self._minmax_sum = _np.round(self.max + self.min, decimals=15)
        return self._minmax_sum

    @property
    def max(self):
        """Get the maximum RGB value for each color."""
        if self._max is None:
            self._max = _np.max(self._rgb[..., :3], axis=-1)
        return self._max

    @property
    def min(self):
        """Get the minimum RGB value for each color."""
        if self._min is None:
            self._min = _np.min(self._rgb[..., :3], axis=-1)
        return self._min


class HXXConverter:
    """Convert HSL, HSV, or HWB colors to other color systems in the RGB model."""

    def __init__(
        self,
        system: _Literal["hsl", "hsv", "hwb"],
        hxx: _Sequence[float | int] | _np.ndarray
    ):
        """
        Parameters
        ----------
        system : {'hsl', 'hsv', 'hwb'}
            Color system of the input values.
        hxx : array of numbers
            Array of HSL, HSV, or HWB values (shape: [..., 3]) with optional alpha (shape: [..., 4]).
        """
        hxx = _np.asarray(hxx)
        color_dim = hxx.shape[-1]
        if color_dim not in (3, 4):
            raise ValueError(f"{system.upper()} values must have 3 or 4 components.")
        self._hue = hxx[..., 0]
        self._hsl_s = hxx[..., 1] if system == "hsl" else None
        self._hsl_l = hxx[..., 2] if system == "hsl" else None
        self._hsv_s = hxx[..., 1] if system == "hsv" else None
        self._hsv_v = hxx[..., 2] if system == "hsv" else None
        self._hwb_w = hxx[..., 1] if system == "hwb" else None
        self._hwb_b = hxx[..., 2] if system == "hwb" else None
        self._alpha = hxx[..., 3] if color_dim == 4 else None
        if system == "hwb":
            wb_sum = self._hwb_w + self._hwb_b
            wb_sum_greater_1 = wb_sum > 1
            self._hwb_w = _np.where(wb_sum_greater_1, self._hwb_w / wb_sum, self._hwb_w)
            self._hwb_b = _np.where(wb_sum_greater_1, self._hwb_b / wb_sum, self._hwb_b)
        return

    def convert(self, target: _Literal["rgb", "hsl", "hsv", "hwb"], decimals: int = 5) -> _np.ndarray:
        """Convert color to RGB, HSL, HSV, or HWB.

        Parameters
        ----------
        target : {'rgb', 'hsl', 'hsv', 'hwb'}
            Target color system to convert the HXX values to.
        decimals : int, default: 5
            Number of decimal places to round the target values to.

        Returns
        -------
        target_colors : numpy.ndarray
            Array of target color system values with the same shape as the input HXX array.
        """
        if target == "rgb":
            return self.to_rgb(decimals)
        if target == "hsl":
            return self.to_hsl(decimals)
        if target == "hsv":
            return self.to_hsv(decimals)
        if target == "hwb":
            return self.to_hwb(decimals)
        raise ValueError(f"Conversion to {target.upper()} is not supported.")

    def to_rgb(self, decimals: int = 5):
        """Convert to RGB.

        Parameters
        ----------
        decimals : int, default: 5
            Number of decimal places to round the RGB values to.

        Returns
        -------
        rgb_colors : numpy.ndarray
            Array of RGB values with the same shape as the input HXX color array.
        """
        m2 = _np.where(
            self.hsl_lightness <= 0.5,
            self.hsl_lightness * (1 + self.hsl_saturation),
            self.hsl_lightness + self.hsl_saturation - (self.hsl_lightness * self.hsl_saturation),
        )
        m1 = 2 * self.hsl_lightness - m2
        m2_minus_m1 = m2 - m1

        components = [self._calc_rgb(self._hue + offset, m1, m2, m2_minus_m1) for offset in (1 / 3, 0, -1 / 3)]
        if self._alpha:
            components.append(self._alpha)
        return _np.round(_np.stack(components, axis=-1), decimals=decimals)

    def _calc_rgb(self, hue, m1, m2, m2_minus_m1):
        hue = _np.mod(hue, 1)
        with _np.errstate(divide='ignore', invalid='ignore'):
            return _np.where(
                self.hsl_saturation == 0,
                self.hsl_lightness,
                _np.where(
                    hue < 1 / 6,
                    m1 + m2_minus_m1 * hue * 6,
                    _np.where(
                        hue < 0.5,
                        m2,
                        _np.where(
                            hue < 2 / 3,
                            m1 + m2_minus_m1 * (2 / 3 - hue) * 6,
                            m1,
                        ),
                    ),
                )
            )

    def to_hsl(self, decimals: int = 5):
        """Convert to HSL.

        Parameters
        ----------
        decimals : int, default: 5
            Number of decimal places to round the HSL values to.

        Returns
        -------
        hsl_colors : numpy.ndarray
            Array of HSL values with the same shape as the input HXX color array.
        """
        components = [self._hue, self.hsl_saturation, self.hsl_lightness]
        if self._alpha:
            components.append(self._alpha)
        return _np.round(_np.stack(components, axis=-1), decimals=decimals)

    def to_hsv(self, decimals: int = 5):
        """Convert to HSV.

        Parameters
        ----------
        decimals : int, default: 5
            Number of decimal places to round the HSV values to.

        Returns
        -------
        hsv_colors : numpy.ndarray
            Array of HSV values with the same shape as the input HXX color array.
        """
        components = [self._hue, self.hsv_saturation, self.hsv_value]
        if self._alpha:
            components.append(self._alpha)
        return _np.round(_np.stack(components, axis=-1), decimals=decimals)

    def to_hwb(self, decimals: int = 5):
        """Convert to HWB.

        Parameters
        ----------
        decimals : int, default: 5
            Number of decimal places to round the HWB values to.

        Returns
        -------
        hwb_colors : numpy.ndarray
            Array of HWB values with the same shape as the input HXX color array.
        """
        components = [self._hue, self.hwb_whiteness, self.hwb_blackness]
        if self._alpha:
            components.append(self._alpha)
        return _np.round(_np.stack(components, axis=-1), decimals=decimals)

    @property
    def hsl_saturation(self):
        """HSL S (saturation) values for each color."""
        if self._hsl_s is not None:
            return self._hsl_s
        return _np.where(
            _np.isin(self.hsl_lightness, [0, 1]),
            0,
            (self.hsv_value - self.hsl_lightness) / _np.minimum(self.hsl_lightness, 1 - self.hsl_lightness),
        )

    @property
    def hsl_lightness(self):
        """HSL L (lightness) values for each color."""
        if self._hsl_l is None:
            self._hsl_l = _np.round(self.hsv_value * (1 - (self.hsv_saturation / 2)), decimals=15)
        return self._hsl_l

    @property
    def hsv_saturation(self):
        """HSV S (saturation) values for each color."""
        if self._hsv_s is not None:
            return self._hsv_s
        if self._hsl_s is not None:
            self._hsv_s = _np.where(
                self.hsv_value == 0,
                0,
                2 * (1 - (self.hsl_lightness / self.hsv_value)),
            )
            return self._hsv_s
        return 1 - (self._hwb_w / self.hsv_value)

    @property
    def hsv_value(self):
        """HSV V (value) values for each color."""
        if self._hsv_v is not None:
            return self._hsv_v
        if self._hsl_s is not None:
            self._hsv_v = self._hsl_s * _np.minimum(self._hsl_l, 1 - self._hsl_l) + self._hsl_l
            return self._hsv_v
        return 1 - self._hwb_b

    @property
    def hwb_whiteness(self):
        """HWB W (whiteness) values for each color."""
        if self._hwb_w is not None:
            return self._hwb_w
        return self.hsv_value * (1 - self.hsv_saturation)

    @property
    def hwb_blackness(self):
        """HWB B (blackness) values for each color."""
        if self._hwb_b is not None:
            return self._hwb_b
        return 1 - self.hsv_value


def convert_color(
    values: _Sequence[float | int] | _np.ndarray,
    source_system: _Literal["rgb", "hsl", "hsv", "hwb"],
    target_system: _Literal["rgb", "hsl", "hsv", "hwb"],
    decimals: int = 5,
) -> _np.ndarray:
    """Convert RGB colors between RGB, HSL, HSV, and HWB coordinate systems.

    Parameters
    ----------
    values : array of numbers
        Array of RGB, HSL, HSV, or HWB colors, with or without alpha channel;
        i.e., shape [..., 3] or [..., 4].
    source_system : {'rgb', 'hsl', 'hsv', 'hwb'}
        Coordinate system of the input values.
    target_system : {'rgb', 'hsl', 'hsv', 'hwb'}
        Coordinate system to convert the input values to.
    decimals : int, default: 5
        Number of decimal places to round the target values to.
    """
    if source_system == "rgb":
        converter = RGBConverter(rgb=values)
    else:
        converter = HXXConverter(system=source_system, hxx=values)
    return converter.convert(target=target_system, decimals=decimals)


def convert_angle(
    value: float,
    source_unit: _Literal["deg", "grad", "rad", "turn"],
    target_unit: _Literal["deg", "grad", "rad", "turn"],
    wrap: bool = False,
) -> float:
    """Convert a [hue angle](https://developer.mozilla.org/en-US/docs/Web/CSS/hue) from one
    [unit](https://developer.mozilla.org/en-US/docs/Web/CSS/angle#units) to another,
    optionally wrapping it within the range of the target unit.

    Parameters
    ----------
    value : float
        Angle value to convert.
    source_unit : {'deg', 'grad', 'rad', 'turn'}
        Unit of the input angle value.
    target_unit : {'deg', 'grad', 'rad', 'turn'}
        Unit of the output angle value.
    wrap : bool, default: False
        Whether to wrap the angle within the range of the target unit.
        This will also make the value positive.

    Returns
    -------
    target_value : float
        Converted angle value.
    """
    circle_parts = {
        "deg": 360,
        "grad": 400,
        "rad": 2 * _np.pi,
        "turn": 1,
    }
    target_value = value * (circle_parts[target_unit] / circle_parts[source_unit])
    if wrap:
        target_value = target_value % circle_parts[target_unit]
    return _number.clean_float(target_value)


