import itertools
import colorsys

import numpy as np
from pycolorit import conversion


def colorsys_comparison_test():
    """Compare conversion results with those calculated by Python's `colorsys` module."""
    for color in np.random.rand(10000, 3):

        # Assume color is RGB and convert to HSL
        hsl_self = conversion.convert_color(
            values=color,
            source_system="rgb",
            target_system="hsl",
            decimals=8
        )
        # colorsys returns in order H, L, S instead of H, S, L
        h, l, s = colorsys.rgb_to_hls(*color)
        assert np.allclose(hsl_self, (h, s, l))

        # Assume color is HSL and convert to RGB
        rgb_self = conversion.convert_color(
            values=color,
            source_system="hsl",
            target_system="rgb",
            decimals=8
        )
        # colorsys accepts in order H, L, S instead of H, S, L
        rgb_colorsys = colorsys.hls_to_rgb(color[0], color[2], color[1])
        assert np.allclose(rgb_self, rgb_colorsys)

        # Assume color is RGB and convert to HSV
        hsv_self = conversion.convert_color(
            values=color,
            source_system="rgb",
            target_system="hsv",
            decimals=8
        )
        hsv_colorsys = colorsys.rgb_to_hsv(*color)
        assert np.allclose(hsv_self, hsv_colorsys)

        # Assume color is HSV and convert to RGB
        rgb_self = conversion.convert_color(
            values=color,
            source_system="hsv",
            target_system="rgb",
            decimals=8
        )
        rgb_colorsys = colorsys.hsv_to_rgb(color[0], color[1], color[2])
        assert np.allclose(rgb_self, rgb_colorsys)
    return


def roundtrip_consistency_test():
    """Test that converting from one system to another and back yields the original color.

    This test is performed for all possible combinations of RGB, HSL, HSV, and HWB color systems.
    """
    for source_system, target_system in itertools.permutations(["rgb", "hsl", "hsv", "hwb"], 2):
        colors = np.random.rand(30000, 3)
        if source_system == "hwb":
            # In HWB colors, if the sum of Whiteness and Blackness is equal to or more than 1,
            # then the color will be a shade of grey, regardless of the hue value.
            # In such cases, the roundtrip will correctly change the hue value to 0.
            # Therefore, we ignore colors where W + B >= 1
            wb_sum = colors[:, 1] + colors[:, 2]
            colors = colors[wb_sum < 1]
        colors_converted = conversion.convert_color(
            values=colors,
            source_system=source_system,
            target_system=target_system,
            decimals=15
        )
        colors_roundtrip = conversion.convert_color(
            values=colors_converted,
            source_system=target_system,
            target_system=source_system,
            decimals=15
        )
        assert np.allclose(colors, colors_roundtrip, rtol=0, atol=1e-7)
    return
