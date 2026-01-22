import os
import numpy as np

from PIL import Image


"""
We can use numpy with PIL to take images and turn them into np arrays for easier calculations.
"""


############################# Main Functions #############################
##########################################################################
def make_g(grayscale_array: np.ndarray):
    """
    Build g : Ω -> R from a grayscale image array.
    Input:
      grayscale_array: (H, W)
    Returned:
      g(x) where x=(x1,x2) in [0,1]^2, returning grayscale_array[row, col]
    """
    grayscale_array = np.asarray(grayscale_array)
    if grayscale_array.ndim != 2:
        raise ValueError(
            f"grayscale_array must have shape (H,W), got {grayscale_array.shape}"
        )

    H, W = grayscale_array.shape

    def g(x: tuple) -> float:
        x1, x2 = x

        # clamp to [0,1] for safety
        x1 = float(np.clip(x1, 0.0, 1.0))
        x2 = float(np.clip(x2, 0.0, 1.0))

        col = min(int(np.floor(x1 * W)), W - 1)
        row = min(int(np.floor(x2 * H)), H - 1)

        return float(grayscale_array[row, col])

    return g


def make_f(
    color_array: np.ndarray,
    mask: np.ndarray,
    *,
    strict_domain: bool = True,
    fill_value=None,
):
    """
    Build f : D -> R^3 from a color image and a boolean mask of known pixels.

    Inputs:
      color_array: (H,W,3) RGB array (uint8 or float)
      mask:        (H,W) boolean, True = known color (in D), False = unknown (outside D)

    Returned function:
      f(x) where x=(x1,x2) in [0,1]^2
        - x1 -> column, x2 -> row
        - maps via j = floor(x1*W), i = floor(x2*H) (with clamping at edges)
        - if mask[i,j] is True, returns RGB at that pixel
        - else:
            * if strict_domain=True: raises ValueError (f undefined outside D)
            * otherwise returns fill_value (default: None)

    Options:
      normalized_output:
        - if True and input is uint8-like, returns float32 RGB in [0,1]
        - if False, returns raw values (e.g. uint8 0..255)
    """
    color_array = np.asarray(color_array)
    mask = np.asarray(mask, dtype=bool)

    if color_array.ndim != 3 or color_array.shape[2] != 3:
        raise ValueError(
            f"color_array must have shape (H,W,3), got {color_array.shape}"
        )

    H, W, _ = color_array.shape
    if mask.shape != (H, W):
        raise ValueError(f"mask must have shape {(H,W)}, got {mask.shape}")

    def f(x: tuple) -> np.ndarray:
        x1, x2 = x

        # clamp x to [0,1] (or change to "raise" if you prefer strict input)
        x1 = float(np.clip(x1, 0.0, 1.0))
        x2 = float(np.clip(x2, 0.0, 1.0))

        j = min(int(np.floor(x1 * W)), W - 1)  # col
        i = min(int(np.floor(x2 * H)), H - 1)  # row

        if not mask[i, j]:
            if strict_domain:
                raise ValueError(
                    f"x maps to unknown pixel (row={i}, col={j}); f is only defined on D."
                )
            return fill_value

        rgb = color_array[i, j]
        return np.asarray(rgb)

    return f


def make_nonlocal_k(g, *, t: float, p: float = 2.0, eps: float = 1e-12):
    """
    Build kernel k(x,y) = exp( - |g(x)-g(y)|^p / (4t) ).

    Args:
      g: function from make_g, mapping x=(x1,x2) in [0,1]^2 -> grayscale scalar
      t: positive scale parameter (t > 0)
      p: exponent, typically in (0, 2]
      eps: tiny constant to avoid division by zero if t is extremely small

    Returns:
      k(x,y) as a float in (0, 1].
    """
    t = float(t)
    p = float(p)
    if t <= 0:
        raise ValueError(f"t must be > 0, got {t}")
    if not 0 < p <= 2:
        raise ValueError(f"p must be in (0,2], got {p}")

    denom = 4.0 * t + eps

    def k(x, y) -> float:
        gx = float(g(x))
        gy = float(g(y))
        diff = abs(gx - gy)
        return float(np.exp(-(diff**p) / denom))

    return k


def make_vector_valued_K(k, *, dim: int = 3, dtype=float):
    """
    Build vector-valued kernel K(x,y) = k(x,y) * I_{dim}.

    Args:
      k: scalar kernel function (e.g., from make_k), k(x,y) -> float
      dim: output dimension (3 for RGB)
      dtype: numpy dtype for the returned matrix

    Returns:
      K(x,y) -> (dim, dim) numpy array (diagonal matrix).
    """
    I = np.eye(dim, dtype=dtype)

    def K(x: tuple, y: tuple) -> np.ndarray:
        s = dtype(k(x, y))
        return s * I  # diagonal matrix with k(x,y) on the diagonal

    return K


if __name__ == "__main__": 
    pass
