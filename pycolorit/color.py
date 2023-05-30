from typing import Sequence
import jax
import numpy as np
import jax.numpy as jnp


class Color:

    def __init__(self, values: np.ndarray):
        self._values = values
        return

    @property
    def rgb(self) -> np.ndarray:
        return self._values

    @property
    def hsl(self):
        norm = self.rgb / 255
        maxes = norm.max(axis=1)
        mins = norm.min(axis=1)
        dists = maxes - mins
        sums = maxes + mins
        # Calculate all 'L' values, and create array with 'H' and 'S' values all set to zero.
        init_hsls = np.pad((maxes + mins) / 2, pad_width=((0, 0), (2, 0)), mode='constant', constant_values=0)
        # For cases where max. values and min. values are not the same, 'H' and 'S' values are non-zero
        #  and must be calculated:
        mask = dists != 0
        sel_dists = dists[mask]
        sel_norm = norm[mask]
        sel_ls = init_hsls[mask, 2]
        sel_sums = sums[mask]
        # Set 'S' values
        init_hsls[mask, 1] = np.where(sel_ls > 0.5, sel_dists / (2 - sel_sums), sel_dists / sel_sums)
        # Set 'H' values
        sel_maxes = maxes[mask]
        init_hsls[mask, 0] = np.where(
            sel_maxes == sel_norm[:, 0],
            (sel_norm[:, 1] - sel_norm[:, 2]) / s
        )
        return

    def hsl2(self):
        return



def rgb(values: tuple[int, int, int] | Sequence[tuple[int, int, int]]):
    colors = np.asarray(values)
    if not np.issubdtype(colors.dtype, np.integer):
        raise ValueError(f"`values` must be a sequence of integers, but found elements with type {colors.dtype}")
    if np.any(np.logical_or(colors < 0, colors > 255)):
        raise ValueError("`values` must be in the range [0, 255].")
    colors = colors.astype(np.ubyte)
    if colors.ndim > 2 or colors.shape[-1] != 3:
        raise ValueError(f"`values` must either have a shape of (3, ) or (n, 3). The input shape was {colors.shape}")
    colors = colors[np.newaxis] if colors.ndim == 1 else colors
    return Color(values=colors)


def hexa(values: str | Sequence[str]) -> Color:

    def process_single_hex(val: str) -> tuple[int, int, int]:
        if len(val) == 3:
            val = ''.join([d * 2 for d in val])
        elif len(val) != 6:
            raise ValueError("Hex color not recognized.")
        return tuple(int(val[i:i + 2], 16) for i in range(0, 5, 2))

    colors = np.asarray(values)
    if not np.issubdtype(colors.dtype, np.str_):
        raise ValueError(f"`values` must be a sequence of strings, but found elements with type {colors.dtype}")
    if colors.ndim == 0:
        colors = colors[np.newaxis]
    elif colors.ndim > 1:
        raise ValueError(
            f"`values` must either be a string, or a 1-dimensional sequence. The input dimension was {colors.ndim}"
        )
    colors = np.char.lstrip('#')
    colors_rgb = np.empty(shape=(colors.size, 3), dtype=np.ubyte)
    for i, color in enumerate(colors):
        colors_rgb[i] = process_single_hex(color)
    return Color(values=colors_rgb)


@jax.jit
def _rgb_to_hsl(colors: jnp.ndarray):
    norm = colors / 255
    max_val = jnp.max(norm, axis=-1)
    min_val = jnp.min(norm, axis=-1)
    minmax_sum = max_val + min_val
    minmax_dist = max_val - min_val
    hsl_l = minmax_sum / 2
    hsl_s = jnp.where(
        hsl_l > 0.5,
        minmax_dist / (2 - minmax_sum),
        minmax_dist / minmax_sum
    )
    hsl_h = jnp.where(
        minmax_dist == 0,
        0,
        jnp.where(
            max_val == norm[..., 0],
            (norm[..., 1] - norm[..., 2]) / minmax_dist + jnp.where(norm[..., 1] < norm[..., 2], 6, 0),
            jnp.where(
                max_val == norm[..., 1],
                (norm[..., 2] - norm[..., 0]) / minmax_dist + 2,
                (norm[..., 0] - norm[..., 1]) / minmax_dist + 4,
            )
        )
    ) / 6
    return jnp.stack([hsl_h, hsl_s, hsl_l], axis=-1) * 100