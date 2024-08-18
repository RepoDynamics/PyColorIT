"""Work with colors in various color spaces and coordinate systems."""

from typing import Literal as _Literal, Sequence as _Sequence
import re as _re

from IPython import display as _display
import numpy as _np

from pycolorit import exception as _exception
from pycolorit import _data, conversion as _conversion, number as _number


class RGBColor:
    """Color in RGB color space."""

    def __init__(
        self,
        system: _Literal["rgb", "hsl", "hsv", "hwb"],
        values: _Sequence[float],
    ):
        self._values = {system: tuple(values)}
        return

    def array(
        self,
        system: _Literal["rgb", "hsl", "hsv", "hwb"],
        hue_unit: _Literal["deg", "grad", "rad", "turn"] = "turn",
        hue_wrap: bool = False,
        explicit_alpha: bool = False,
        ubyte: bool = False,
    ):
        """Get the color component values in the specified coordinate system.

        Parameters
        ----------
        system : {'rgb', 'hsl', 'hsv', 'hwb'}
            Coordinate system to return the color values in.
        hue_unit : {'deg', 'grad', 'rad', 'turn'}, default: 'turn'
            Unit of the hue component.
            This is only applicable to HSL, HSV, and HWB systems.
        hue_wrap : bool, default: False
            Only applicable to HSL, HSV, and HWB systems:
            If True, the hue value is wrapped within the range of the unit.
        explicit_alpha : bool, default: False
            If True, an alpha channel is added if it is not already present.
        ubyte : bool, default: False
            If True, the values are returned as unsigned bytes (uint8) in
            the range [0, 255] instead of the default [0, 1].
            This is only applicable to RGB values.

        Returns
        -------
        numpy.ndarray
            Array of color component values in the specified system.
        """
        if system == "rgb":
            return self.rgb(ubyte=ubyte, explicit_alpha=explicit_alpha)
        if system == "hsl":
            return self.hsl(hue_unit=hue_unit, hue_wrap=hue_wrap, explicit_alpha=explicit_alpha)
        if system == "hsv":
            return self.hsv(hue_unit=hue_unit, hue_wrap=hue_wrap, explicit_alpha=explicit_alpha)
        if system == "hwb":
            return self.hwb(hue_unit=hue_unit, hue_wrap=hue_wrap, explicit_alpha=explicit_alpha)
        raise ValueError(f"Invalid system '{system}'.")

    def css(
        self,
        system: _Literal["hex", "rgb", "hsl", "hwb"],
        decimals: int = 2,
        percent: bool = False,
        hue_unit: _Literal["deg", "grad", "rad", "turn"] = "turn",
        hue_wrap: bool = False,
        explicit_alpha: bool = False,
    ) -> str:
        """Get the color as a CSS color value.

        Parameters
        ----------
        system : {'hex', 'rgb', 'hsl', 'hwb'}
            CSS color format to return the color in.
        decimals : int, default: 2
            Number of decimal places to round the color components to.
            This is not applicable to the HEX system.
        percent : bool, default: False
            If True, the color components that
            can be represented as percentages are shown as such.
        hue_unit : {'deg', 'grad', 'rad', 'turn'}, default: 'turn'
            Unit of the hue component.
            This is only applicable to HSL, HSV, and HWB systems.
        hue_wrap : bool, default: False
            Only applicable to HSL, HSV, and HWB systems:
            If True, the hue value is wrapped within the range of the unit.
        explicit_alpha : bool, default: False
            If True, an alpha channel is added if it is not already present.

        Returns
        -------
        str
            CSS color value in the specified format.
        """
        if system == "hex":
            return self.css_hex(explicit_alpha=explicit_alpha)
        if system == "rgb":
            return self.css_rgb(decimals=decimals, percent=percent, explicit_alpha=explicit_alpha)
        if system not in ("hsl", "hwb"):
            raise ValueError(f"Invalid system '{system}'.")
        return self._css_hxx(
            system=system,
            decimals=decimals,
            percent=percent,
            hue_unit=hue_unit,
            hue_wrap=hue_wrap,
            explicit_alpha=explicit_alpha,
        )

    def rgb(self, ubyte: bool = False, explicit_alpha: bool = False) -> _np.ndarray:
        """Get the color as an RGB(A) array.

        Parameters
        ----------
        ubyte : bool, default: False
            If True, the values are returned as unsigned bytes (uint8) in
            the range [0, 255] instead of the default [0, 1].
        explicit_alpha : bool, default: False
            If True, an alpha channel is added if it is not already present.

        Returns
        -------
        numpy.ndarray
            Array of [R, G, B, (A)] color components.
        """
        rgb = self._get_or_convert("rgb")
        if explicit_alpha and len(rgb) == 3:
            rgb = (*rgb, 1)
        rgb_array = _np.array(rgb)
        return _np.round(rgb_array * 255).astype(_np.ubyte) if ubyte else rgb_array

    def hsl(
        self,
        hue_unit: _Literal["deg", "grad", "rad", "turn"] = "turn",
        hue_wrap: bool = False,
        explicit_alpha: bool = False
    ) -> _np.ndarray:
        """Get the color as an HSL(A) array.

        Parameters
        ----------
        hue_unit : {'deg', 'grad', 'rad', 'turn'}, default: 'turn'
            Unit of the hue component.
        hue_wrap : bool, default: False
            If True, the hue value is wrapped within the range of the unit.
        explicit_alpha : bool, default: False
            If True, an alpha channel is added if it is not already present.

        Returns
        -------
        numpy.ndarray
            Array of [H, S, L, (A)] color components.
        """
        return self._hxx("hsl", hue_unit, hue_wrap, explicit_alpha)

    def hsv(
        self,
        hue_unit: _Literal["deg", "grad", "rad", "turn"] = "turn",
        hue_wrap: bool = False,
        explicit_alpha: bool = False
    ) -> _np.ndarray:
        """Get the color as an HSV(A) array.

        Parameters
        ----------
        hue_unit : {'deg', 'grad', 'rad', 'turn'}, default: 'turn'
            Unit of the hue component.
        hue_wrap : bool, default: False
            If True, the hue value is wrapped within the range of the unit.
        explicit_alpha : bool, default: False
            If True, an alpha channel is added if it is not already present.

        Returns
        -------
        numpy.ndarray
            Array of [H, S, V, (A)] color components.
        """
        return self._hxx("hsv", hue_unit, hue_wrap, explicit_alpha)

    def hwb(
        self,
        hue_unit: _Literal["deg", "grad", "rad", "turn"] = "turn",
        hue_wrap: bool = False,
        explicit_alpha: bool = False
    ) -> _np.ndarray:
        """Get the color as an HWB(A) array.

        Parameters
        ----------
        hue_unit : {'deg', 'grad', 'rad', 'turn'}, default: 'turn'
            Unit of the hue component.
        hue_wrap : bool, default: False
            If True, the hue value is wrapped within the range of the unit.
        explicit_alpha : bool, default: False
            If True, an alpha channel is added if it is not already present.

        Returns
        -------
        numpy.ndarray
            Array of [H, W, B, (A)] color components.
        """
        return self._hxx("hwb", hue_unit, hue_wrap, explicit_alpha)

    def css_hex(self, explicit_alpha: bool = False) -> str:
        """Get the color as a CSS hexadecimal color value.

        Parameters
        ----------
        explicit_alpha : bool, default: False
            If True, an alpha channel is added if it is not already present.

        Returns
        -------
        str
            CSS hexadecimal color value, e.g., '#FF001DAA'.
        """
        hex_str = "".join(f"{c:02X}" for c in self.rgb(ubyte=True, explicit_alpha=explicit_alpha))
        return f"#{hex_str}"

    def css_rgb(self, decimals: int = 2, percent: bool = False, explicit_alpha: bool = False) -> str:
        """Get the color as a CSS `rgb()` color value.

        Parameters
        ----------
        decimals : int, default: 2
            Number of decimal places to round the color components to.
        percent : bool, default: False
            If True, the color components are shown as percentages.
        explicit_alpha : bool, default: False
            If True, an alpha channel is added if it is not already present.

        Returns
        -------
        str
            CSS `rgb()` color value, e.g., 'rgb(255 0 29)', 'rgb(100% 0% 11% / 50%)'.
        """
        rgb = self.rgb(explicit_alpha=explicit_alpha)
        rgb[:(4 if percent else 3)] *= 100 if percent else 255
        rgb = [_number.clean_float(c) for c in _np.round(rgb, decimals=decimals)]
        css_rgb = [f"{c}{'%' if percent else ''}" for c in rgb[:3]]
        if len(rgb) == 4:
            css_rgb.extend(["/", f"{rgb[3]}{'%' if percent else ''}"])
        return f"rgb({" ".join(css_rgb)})"

    def css_hsl(
        self,
        decimals: int = 2,
        percent: bool = False,
        hue_unit: _Literal["deg", "grad", "rad", "turn"] = "turn",
        hue_wrap: bool = False,
        explicit_alpha: bool = False,
    ) -> str:
        """Get the color as a CSS `hsl()` color value.

        Parameters
        ----------
        decimals : int, default: 2
            Number of decimal places to round the color components to.
        percent : bool, default: False
            If True, the color components are shown as percentages.
        hue_unit : {'deg', 'grad', 'rad', 'turn'}, default: 'turn'
            Unit of the hue component.
        hue_wrap : bool, default: False
            If True, the hue value is wrapped within the range of the unit.
        explicit_alpha : bool, default: False
            If True, an alpha channel is added if it is not already present.

        Returns
        -------
        str
            CSS `hsl()` color value, e.g., 'hsl(0 100% 50%)', 'hsl(0deg 100% 50% / 50%)'.
        """
        return self._css_hxx(
            system="hsl",
            decimals=decimals,
            percent=percent,
            hue_unit=hue_unit,
            hue_wrap=hue_wrap,
            explicit_alpha=explicit_alpha,
        )

    def css_hwb(
        self,
        decimals: int = 2,
        percent: bool = False,
        hue_unit: _Literal["deg", "grad", "rad", "turn"] = "turn",
        hue_wrap: bool = False,
        explicit_alpha: bool = False,
    ) -> str:
        """Get the color as a CSS `hwb()` color value.

        Parameters
        ----------
        decimals : int, default: 2
            Number of decimal places to round the color components to.
        percent : bool, default: False
            If True, the color components are shown as percentages.
        hue_unit : {'deg', 'grad', 'rad', 'turn'}, default: 'turn'
            Unit of the hue component.
        hue_wrap : bool, default: False
            If True, the hue value is wrapped within the range of the unit.
        explicit_alpha : bool, default: False

        Returns
        -------
        str
            CSS `hwb()` color value, e.g., 'hwb(0 100% 0%)', 'hwb(0deg 100% 0% / 50%)'.
        """
        return self._css_hxx(
            system="hwb",
            decimals=decimals,
            percent=percent,
            hue_unit=hue_unit,
            hue_wrap=hue_wrap,
            explicit_alpha=explicit_alpha,
        )

    def display(
        self,
        size: int = 100,
        system: _Literal["hex", "rgb", "hsl", "hwb"] = "hex",
        decimals: int = 2,
        percent: bool = False,
        hue_unit: _Literal["deg", "grad", "rad", "turn"] = "turn",
        hue_wrap: bool = False,
        explicit_alpha: bool = False,
    ):
        """Display the color as a rectangle in the Jupyter notebook.

        Parameters
        ----------
        size : int, default: 100
            Size of the square rectangle in pixels.
        system : {'hex', 'rgb', 'hsl', 'hwb'}, default: 'hex'
            Coordinate system to display the color in.
        decimals : int, default: 2
            Number of decimal places to round the color components to.
        percent : bool, default: False
            If True, the color components are shown as percentages.
        hue_unit : {'deg', 'grad', 'rad', 'turn'}, default: 'turn'
            Unit of the hue component.
        hue_wrap : bool, default: False
            If True, the hue value is wrapped within the range of the unit.
        explicit_alpha : bool, default: False
            If True, an alpha channel is added if it is not already present.
        """
        css_color = self.css(
            system=system,
            decimals=decimals,
            percent=percent,
            hue_unit=hue_unit,
            hue_wrap=hue_wrap,
            explicit_alpha=explicit_alpha,
        )
        svg = (
            f'<svg width="{size}" height="{size}">'
            f'<rect width="{size}" height="{size}" '
            f'style="fill:{css_color}; stroke-width:2; stroke:#000" />'
            '</svg>'
        )
        _display.display(_display.HTML(svg))
        return

    def _get_or_convert(self, system: _Literal["rgb", "hsl", "hsv", "hwb"]) -> _np.ndarray:
        """Get the color values in the specified system if already available,
        or calculate them from another available system."""
        if system in self._values:
            return _np.array(self._values[system])
        systems = ["rgb", "hsl", "hsv", "hwb"]
        systems.remove(system)
        for rem_system in systems:
            if rem_system in self._values:
                new_values = _conversion.convert_color(
                    self._values[rem_system], source_system=rem_system, target_system=system
                )
                self._values[system] = tuple(new_values)
                return new_values
        raise ValueError(f"Conversion to {system} is not supported.")

    def _hxx(
        self,
        system: _Literal["hsl", "hsv", "hwb"],
        hue_unit: _Literal["deg", "grad", "rad", "turn"] = "turn",
        hue_wrap: bool = False,
        explicit_alpha: bool = False
    ) -> _np.ndarray:
        """Get the color as an HXX(A) array, i.e., HSL, HSV or HWB."""
        hxx = self._get_or_convert(system)
        if explicit_alpha and len(hxx) == 3:
            hxx = (*hxx, 1)
        hxx_array = _np.array(hxx)
        hxx_array[0] = _conversion.convert_angle(
            value=hxx_array[0],
            source_unit="turn",
            target_unit=hue_unit,
            wrap=hue_wrap,
        )
        return hxx_array

    def _css_hxx(
        self,
        system: _Literal["hsl", "hwb"],
        decimals: int,
        percent: bool,
        hue_unit: _Literal["deg", "grad", "rad", "turn"],
        hue_wrap: bool,
        explicit_alpha: bool,
    ):
        """Get the color as a CSS `hxx()` color value,
        i.e., `hsl()` or `hwb()`."""
        hxx_func = self.hsl if system == "hsl" else self.hwb
        hxx = hxx_func(hue_unit=hue_unit, hue_wrap=hue_wrap, explicit_alpha=explicit_alpha)
        hxx[1:(4 if percent else 3)] *= 100
        hxx = [_number.clean_float(c) for c in _np.round(hxx, decimals=decimals)]
        css_hxx = [f"{hxx[0]}{hue_unit}"] + [f"{c}%" for c in hxx[1:3]]
        if len(hxx) == 4:
            css_hxx.extend(["/", f"{hxx[3]}{'%' if percent else ''}"])
        return f"{system}({" ".join(css_hxx)})"


def rgb(values: _Sequence[float], unit: float | _Sequence[float] = 1) -> RGBColor:
    """Create an RGB color from RGB(A) values.

    Parameters
    ----------
    values : Sequence of 3 (R, G, B) or 4 (R, G, B, A) numbers
        Red, Green, Blue, and (optionally) Alpha components of the color.
    unit : number or sequence of 3 or 4 numbers, default: 1
        Maximum value of the color components;
        either one value for all components or one value per component.
        The default is 1, meaning that all values are expected in the range [0, 1].
        For example, if the RGB color is in 8-bit format but the alpha value is in percent,
        then this should be `[255, 255, 255, 100]`.
    """
    return RGBColor(system="rgb", values=tuple(_np.asarray(values) / unit))


def hsl(values: _Sequence[float], unit: float | _Sequence[float] = 1) -> RGBColor:
    """Create an RGB color from HSL(A) values.

    Parameters
    ----------
    values : Sequence of 3 (H, S, L) or 4 (H, S, L, A) numbers
        Hue, Saturation, Lightness, and (optionally) Alpha components of the color.
    unit : float, default: 1
        Maximum value of the color components;
        either one value for all components or one value per component.
        The default is 1, meaning that all values are expected in the range [0, 1].
        For example, for an HSL color where the hue is in degrees and all other components
        are in percent, this should be `[360, 100, 100, 100]`.
        Note that since hue is an angle, unlike other components, its value is unbound.
    """
    return RGBColor(system="hsl", values=tuple(_np.asarray(values) / unit))


def hsv(values: _Sequence[float], unit: float | _Sequence[float] = 1) -> RGBColor:
    """Create an RGB color from HSV(A) values.

    Parameters
    ----------
    values : Sequence of 3 (H, S, V) or 4 (H, S, V, A) numbers
        Hue, Saturation, Value, and (optionally) Alpha components of the color.
    unit : float, default: 1
        Maximum value of the color components;
        either one value for all components or one value per component.
        The default is 1, meaning that all values are expected in the range [0, 1].
        For example, for an HSV color where the hue is in degrees and all other components
        are in percent, this should be `[360, 100, 100, 100]`.
        Note that since hue is an angle, unlike other components, its value is unbound.
    """
    return RGBColor(system="hsv", values=tuple(_np.asarray(values) / unit))


def hwb(values: _Sequence[float], unit: float | _Sequence[float] = 1) -> RGBColor:
    """Create an RGB color from HWB(A) values.

    Parameters
    ----------
    values : Sequence of 3 (H, W, B) or 4 (H, W, B, A) numbers
        Hue, Whiteness, Blackness, and (optionally) Alpha components of the color.
    unit : float, default: 1
        Maximum value of the color components;
        either one value for all components or one value per component.
        The default is 1, meaning that all values are expected in the range [0, 1].
        For example, for an HWB color where the hue is in degrees and all other components
        are in percent, this should be `[360, 100, 100, 100]`.
        Note that since hue is an angle, unlike other components, its value is unbound.
    """
    return RGBColor(system="hwb", values=tuple(_np.asarray(values) / unit))


def css(css_color: str) -> RGBColor:
    """Create an RGB color from a CSS
    [`<color>`](https://developer.mozilla.org/en-US/docs/Web/CSS/color_value) value in sRGB space.

    Parameters
    ----------
    css_color : string
        A CSS color data type in the sRGB model, i.e., one of:
        - [`<named-color>`](https://developer.mozilla.org/en-US/docs/Web/CSS/named-color): 'red', 'blue', 'green', etc.
        - [`<hex-color>`](https://developer.mozilla.org/en-US/docs/Web/CSS/hex-color): '#f00', '#ff0000', '#f00f', '#FF00BBAA', etc.
        - [`rgb()`](https://developer.mozilla.org/en-US/docs/Web/CSS/color_value/rgb): 'rgb(255 0 0)', 'rgb(100% 0% 0%)', etc.
        - [`hsl()`](https://developer.mozilla.org/en-US/docs/Web/CSS/color_value/hsl): 'hsl(0 100% 50%)', 'hsl(0deg 100% 50%)', etc.
        - [`hwb()`](https://developer.mozilla.org/en-US/docs/Web/CSS/color_value/hwb): 'hwb(0 100% 0%)', 'hwb(0deg 100% 0%)', etc.
    """
    css_color = css_color.strip()
    if css_color.startswith("#"):
        return css_hex(css_color)
    if css_color.startswith("rgb"):
        return css_rgb(css_color)
    if css_color.startswith("hsl"):
        return css_hsl(css_color)
    return css_name(css_color)


def css_name(named_color: str) -> RGBColor:
    """Create an `RGBColor` from a CSS
    [`<named-color>`](https://developer.mozilla.org/en-US/docs/Web/CSS/named-color) value.

    Parameters
    ----------
    named_color : string
        A CSS named color, e.g., 'red', 'blue', 'green'.
        You can get a dictionary mapping from all CSS color names to their respective `<hex-color>`
        from `pycolorit.data.css_named_colors()`.
    """
    named_color = named_color.strip()
    css_colors = _data.css_named_colors()
    if named_color not in css_colors:
        raise _exception.PyColorITParseError(
            value=named_color, message=f"CSS color name '{named_color}' not valid."
        )
    return css_hex(css_colors[named_color])


def css_hex(hex_color: str) -> RGBColor:
    """Create an `RGBColor` from a CSS
    [`<hex-color>`](https://developer.mozilla.org/en-US/docs/Web/CSS/hex-color) value.

    Parameters
    ----------
    hex_color : string
        A CSS hexadecimal color value, e.g., '#f00', '#ff0000', '#f00f', '#FF00BBAA'.
    """
    hex_color = hex_color.strip()
    regex = _re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
    if not regex.match(hex_color):
        raise _exception.PyColorITParseError(
            value=hex_color, message=f"Hex color does not match the regex '{regex.pattern}'."
        )
    hex_color = hex_color.removeprefix("#")
    if len(hex_color) in (3, 4):
        hex_color = "".join([d * 2 for d in hex_color])
    rgb = tuple(int(hex_color[i:i + 2], 16) / 255 for i in range(0, len(hex_color) - 1, 2))
    return RGBColor(system="rgb", values=rgb)


def css_rgb(rgb_color: str):
    return _parse_css_color(color=rgb_color, system="rgb")


def css_hsl(hsl_color: str):
    return _parse_css_color(color=hsl_color, system="hsl")


def css_hwb(hwb_color: str):
    return _parse_css_color(color=hwb_color, system="hwb")


def display(
    colors: list[RGBColor],
    width: int = 10,
    height: int = 100,
    gap: int = 0,
    column: bool = False,
    system: _Literal["hex", "rgb", "hsl", "hwb"] = "hex",
    decimals: int = 2,
    percent: bool = False,
    hue_unit: _Literal["deg", "grad", "rad", "turn"] = "turn",
    hue_wrap: bool = False,
    explicit_alpha: bool = False,
):
    """Display a list of RGB colors in Jupyter notebook.

    Parameters
    ----------
    colors : list of RGBColor
        List of RGB colors to display.
    width : int, default: 10
        Width of each color rectangle in pixels.
    height : int, default: 100
        Height of each color rectangle in pixels.
    gap : int, default: 0
        Gap between each color rectangle in pixels.
    column : bool, default: False
        If True, the colors are displayed in a column; otherwise, in a row.
    system : {'hex', 'rgb', 'hsl', 'hwb'}, default: 'hex'
        Coordinate system to display the color in.
    decimals : int, default: 2
        Number of decimal places to round the color components to.
        This is not applicable to the HEX system.
    percent : bool, default: False
        If True, the color components that
        can be represented as percentages are shown as such.
    hue_unit : {'deg', 'grad', 'rad', 'turn'}, default: 'turn'
        Unit of the hue component.
        This is only applicable to HSL, HSV, and HWB systems.
    hue_wrap : bool, default: False
        Only applicable to HSL, HSV, and HWB systems:
        If True, the hue value is wrapped within the range of the unit.
    explicit_alpha : bool, default: False
        If True, an alpha channel is added if it is not already present.
    """
    svgs = []
    for color in colors:
        css_color = color.css(
            system=system,
            decimals=decimals,
            percent=percent,
            hue_unit=hue_unit,
            hue_wrap=hue_wrap,
            explicit_alpha=explicit_alpha,
        )
        svg = (
            f'<svg width="{width}" height="{height}">'
            f'<rect width="{width}" height="{height}" style="fill:{css_color}" />'
            '</svg>'
        )
        svgs.append(svg)
    div_content = f"".join(svgs)
    div_tag = f'<div style="display:flex; flex-direction:{'column' if column else 'row'}; gap:{gap}px">'
    _display.display(_display.HTML(f'{div_tag}{div_content}</div>'))
    return


def _parse_css_color(color: str, system: _Literal["rgb", "hsl", "hwb"]) -> RGBColor:
    """Parse a CSS color value and return an `RGBColor` object."""
    color = color.strip()
    patterns = _RE_CSS_FUNC[system]
    for pattern in patterns:
        match = pattern.match(color)
        if match:
            components = match.groupdict()
            break
    else:
        raise _exception.PyColorITParseError(
            value=color,
            message=(
                f"{system.upper()} color does not match any of the regex patterns: "
                f"{'\n\n'.join(pattern.pattern for pattern in patterns)}"
            ),
        )
    final = []
    if system != "rgb":
        h_unit = components["h_unit"] or "deg"
        h_str = components["h"].removesuffix(h_unit)
        h = _conversion.convert_angle(
            value=0 if h_str == "none" else float(h_str),
            source_unit=h_unit,
            target_unit="turn"
        )
        final.append(h)
    components_and_max = {
        "rgb": (("r", "g", "b"), 255),
        "hsl": (("s", "l"), 100),
        "hwb": (("w", "b"), 100),
    }
    component_keys, max_val_when_not_percent = components_and_max[system]
    for component_key in component_keys:
        component = components[component_key]
        if component == "none":
            component_num = 0
        elif component.endswith("%"):
            component_num = float(component.removesuffix("%")) / 100
        else:
            component_num = float(component) / max_val_when_not_percent
        final.append(min(max(component_num, 0), 1))
    a_str = components["a"]
    if a_str is not None and a_str != "none":
        if a_str.endswith("%"):
            a_num = float(a_str.removesuffix("%")) / 100
        else:
            # Alpha, when not a percentage, is a value between [0, 1]
            a_num = float(a_str)
        final.append(min(max(a_num, 0), 1))
    return RGBColor(system=system, values=final)


_RE_CSS_NUM_UNSIGNED = r"\d*\.?\d+(?:[eE][+-]?\d+)?"
"""Regular expression pattern for an unsigned CSS number."""

_RE_CSS_NUM = rf"[+-]?{_RE_CSS_NUM_UNSIGNED}"
"""Regular expression pattern for a CSS
[`<number>`](https://developer.mozilla.org/en-US/docs/Web/CSS/number)."""

_RE_CSS_NUM_POS = rf"[+]?{_RE_CSS_NUM_UNSIGNED}"
"""Regular expression pattern for a positive CSS number."""

_RE_CSS_PERCENT_POS = rf"{_RE_CSS_NUM_POS}%"
"""Regular expression pattern for a positive CSS
[`<percentage>`](https://developer.mozilla.org/en-US/docs/Web/CSS/percentage)."""

_RE_CSS_ANGLE_UNIT = r"deg|rad|grad|turn"
"""Regular expression pattern for a CSS
[`<angle>`](https://developer.mozilla.org/en-US/docs/Web/CSS/angle) unit."""

_RE_CSS_RGB_NEW = _re.compile(
    rf"""
        ^rgba?\(
            \s*
            (?P<r>({_RE_CSS_NUM_POS}%?)|(none))
            \s*
            (?P<g>({_RE_CSS_NUM_POS}%?)|(none))
            \s*
            (?P<b>({_RE_CSS_NUM_POS}%?)|(none))
            \s*
            (/\s*
            (?P<a>({_RE_CSS_NUM_POS}%?)|(none))
            \s*)?
        \)$
    """,
    _re.VERBOSE
)
"""Regular expression pattern for a CSS
[`<rgb()>`](https://developer.mozilla.org/en-US/docs/Web/CSS/color_value/rgb#formal_syntax) color."""

_RE_CSS_RGB_OLD_NUMBER = _re.compile(
    rf"""
        ^rgba?\(
            \s*
            (?P<r>{_RE_CSS_NUM_POS})
            \s*,\s*
            (?P<g>{_RE_CSS_NUM_POS})
            \s*,\s*
            (?P<b>{_RE_CSS_NUM_POS})
            \s*
            (,\s*
            (?P<a>{_RE_CSS_NUM_POS}%?)
            \s*)?
        \)$
    """,
    _re.VERBOSE
)

_RE_CSS_RGB_OLD_PERCENT = _re.compile(
    rf"""
        ^rgba?\(
            \s*
            (?P<r>{_RE_CSS_NUM_POS}%)
            \s*,\s*
            (?P<g>{_RE_CSS_NUM_POS}%)
            \s*,\s*
            (?P<b>{_RE_CSS_NUM_POS}%)
            \s*
            (,\s*
            (?P<a>{_RE_CSS_NUM_POS}%?)
            \s*)?
        \)$
    """,
    _re.VERBOSE
)

_RE_CSS_HSL_NEW = _re.compile(
    rf"""
        ^hsla?\(
            \s*
            (?P<h>(({_RE_CSS_NUM})(?P<h_unit>{_RE_CSS_ANGLE_UNIT})?)|(none))
            \s*
            (?P<s>({_RE_CSS_NUM_POS}%?)|(none))
            \s*
            (?P<l>({_RE_CSS_NUM_POS}%?)|(none))
            \s*
            (/\s*
            (?P<a>({_RE_CSS_NUM_POS}%?)|(none))
            \s*)?
        \)$
    """,
    _re.VERBOSE
)
"""Regular expression pattern for a CSS
[`<hsl()>`](https://developer.mozilla.org/en-US/docs/Web/CSS/color_value/hsl#formal_syntax) color."""

_RE_CSS_HSL_OLD = _re.compile(
    rf"""
        ^hsla?\(
            \s*
            (?P<h>({_RE_CSS_NUM})(?P<h_unit>{_RE_CSS_ANGLE_UNIT})?)
            \s*,\s*
            (?P<s>{_RE_CSS_NUM_POS}%)
            \s*,\s*
            (?P<l>{_RE_CSS_NUM_POS}%)
            \s*
            (,\s*
            (?P<a>{_RE_CSS_NUM_POS}%?)
            \s*)?
        \)$
    """,
    _re.VERBOSE
)

_RE_CSS_HWB = _re.compile(
    rf"""
        ^hwb\(
            \s*
            (?P<h>(({_RE_CSS_NUM})(?P<h_unit>{_RE_CSS_ANGLE_UNIT})?)|(none))
            \s*
            (?P<w>({_RE_CSS_NUM_POS}%?)|(none))
            \s*
            (?P<b>({_RE_CSS_NUM_POS}%?)|(none))
            \s*
            (/\s*
            (?P<a>({_RE_CSS_NUM_POS}%?)|(none))
            \s*)?
        \)$
    """,
    _re.VERBOSE
)
"""Regular expression pattern for a CSS
[`<hwb()>`](https://developer.mozilla.org/en-US/docs/Web/CSS/color_value/hwb#formal_syntax) color."""

_RE_CSS_FUNC = {
    "rgb": [_RE_CSS_RGB_NEW, _RE_CSS_RGB_OLD_NUMBER, _RE_CSS_RGB_OLD_PERCENT],
    "hsl": [_RE_CSS_HSL_NEW, _RE_CSS_HSL_OLD],
    "hwb": [_RE_CSS_HWB],
}
"""Dictionary of regular expression patterns for CSS functional notations."""
