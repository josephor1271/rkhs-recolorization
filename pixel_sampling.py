import numpy as np

def _validate_img_array(img: np.ndarray) -> tuple[int, int, int | None]:
    img = np.asarray(img)
    if img.ndim == 2:
        H, W = img.shape
        return H, W, None
    if img.ndim == 3:
        H, W, C = img.shape
        return H, W, C
    raise ValueError(f"Expected img with shape (H,W) or (H,W,C), got {img.shape}")

def apply_mask_with_fill(img: np.ndarray, mask: np.ndarray, fill_value=-1) -> np.ndarray:
    """
    img: (H,W) or (H,W,C)
    mask: (H,W) boolean, True = keep, False = unknown
    returns: same shape as img, unknown entries filled with fill_value
    """
    img = np.asarray(img)
    mask = np.asarray(mask, dtype=bool)
    H, W, _ = _validate_img_array(img)
    if mask.shape != (H, W):
        raise ValueError(f"mask must have shape {(H,W)}, got {mask.shape}")

    out = img.copy()
    if img.ndim == 2:
        out[~mask] = fill_value
    else:
        out[~mask, :] = fill_value
    return out

def sample_random_pixels(img: np.ndarray, n: int, *, rng=None) -> np.ndarray:
    """
    Returns a (H,W) boolean mask of selected pixels.
    """
    img = np.asarray(img)
    H, W, _ = _validate_img_array(img)

    n = int(n)
    if n < 0 or n > H * W:
        raise ValueError(f"n must be in [0, {H*W}]")

    rng = np.random.default_rng(rng)
    idx = rng.choice(H * W, size=n, replace=False)

    mask = np.zeros((H, W), dtype=bool)
    r, c = np.divmod(idx, W)
    mask[r, c] = True
    return mask

def sample_random_strips(img: np.ndarray,
                         num_strips: int,
                         *,
                         orientation: str = "both",
                         width_range=(5, 30),
                         rng=None) -> np.ndarray:
    """
    Returns (H,W) boolean mask selecting random horizontal/vertical strips.
    orientation: "horizontal", "vertical", or "both"
    width_range: inclusive-ish pixel widths (min_w, max_w)
    """
    img = np.asarray(img)
    H, W, _ = _validate_img_array(img)

    rng = np.random.default_rng(rng)
    min_w, max_w = map(int, width_range)
    if min_w <= 0 or max_w < min_w:
        raise ValueError("width_range must be (min_w>0, max_w>=min_w)")

    mask = np.zeros((H, W), dtype=bool)

    for _ in range(int(num_strips)):
        if orientation == "both":
            o = "horizontal" if rng.random() < 0.5 else "vertical"
        else:
            o = orientation

        if o == "horizontal":
            w = rng.integers(min_w, max_w + 1)
            r0 = rng.integers(0, H)
            r1 = min(H, r0 + w)
            mask[r0:r1, :] = True
        elif o == "vertical":
            w = rng.integers(min_w, max_w + 1)
            c0 = rng.integers(0, W)
            c1 = min(W, c0 + w)
            mask[:, c0:c1] = True
        else:
            raise ValueError("orientation must be 'horizontal', 'vertical', or 'both'")

    return mask

def sample_random_blotches(img: np.ndarray,
                           num_blotches: int,
                           *,
                           radius_range=(4, 15),
                           rng=None) -> np.ndarray:
    """
    Returns (H,W) boolean mask selecting random circular blotches.
    radius_range: (min_r, max_r) in pixels
    """
    img = np.asarray(img)
    H, W, _ = _validate_img_array(img)

    rng = np.random.default_rng(rng)
    min_r, max_r = map(int, radius_range)
    if min_r <= 0 or max_r < min_r:
        raise ValueError("radius_range must be (min_r>0, max_r>=min_r)")

    mask = np.zeros((H, W), dtype=bool)
    rr = np.arange(H)[:, None]
    cc = np.arange(W)[None, :]

    for _ in range(int(num_blotches)):
        r0 = rng.integers(0, H)
        c0 = rng.integers(0, W)
        rad = rng.integers(min_r, max_r + 1)
        mask |= ((rr - r0) ** 2 + (cc - c0) ** 2) <= rad * rad

    return mask