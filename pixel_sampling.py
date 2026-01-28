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


def sample_random_pixels(n: int, *, rng=None):
    """
    Returns a function sample(img) -> (H,W) boolean mask of n selected pixels.
    rng can be an int seed, a Generator, or None.
    """
    n_fixed = int(n)
    rng_seed = rng

    def sample(img):
        img = np.asarray(img)
        H, W, _ = _validate_img_array(img)

        if n_fixed < 0 or n_fixed > H * W:
            raise ValueError(f"n must be in [0, {H*W}]")

        rng_local = np.random.default_rng(rng_seed)
        idx = rng_local.choice(H * W, size=n_fixed, replace=False)

        mask = np.zeros((H, W), dtype=bool)
        r, c = np.divmod(idx, W)
        mask[r, c] = True
        return mask

    return sample


def sample_random_strips(
    num_strips: int, *, orientation: str = "both", width_range=(5, 30), rng=None
):
    """
    Returns a function sample(img) -> (H,W) boolean mask selecting random strips.
    """
    num_strips_fixed = int(num_strips)
    orientation_fixed = orientation
    width_range_fixed = width_range
    rng_seed = rng

    def sample(img):
        img = np.asarray(img)
        H, W, _ = _validate_img_array(img)

        rng_local = np.random.default_rng(rng_seed)
        min_w, max_w = map(int, width_range_fixed)
        if min_w <= 0 or max_w < min_w:
            raise ValueError("width_range must be (min_w>0, max_w>=min_w)")

        mask = np.zeros((H, W), dtype=bool)

        for _ in range(num_strips_fixed):
            if orientation_fixed == "both":
                o = "horizontal" if rng_local.random() < 0.5 else "vertical"
            else:
                o = orientation_fixed

            if o == "horizontal":
                w = rng_local.integers(min_w, max_w + 1)
                r0 = rng_local.integers(0, H)
                r1 = min(H, r0 + w)
                mask[r0:r1, :] = True
            elif o == "vertical":
                w = rng_local.integers(min_w, max_w + 1)
                c0 = rng_local.integers(0, W)
                c1 = min(W, c0 + w)
                mask[:, c0:c1] = True
            else:
                raise ValueError(
                    "orientation must be 'horizontal', 'vertical', or 'both'"
                )

        return mask

    return sample


def sample_random_blotches(num_blotches: int, *, radius_range=(4, 15), rng=None):
    """
    Returns a function sample(img) -> (H,W) boolean mask selecting random circular blotches.
    """
    num_blotches_fixed = int(num_blotches)
    radius_range_fixed = radius_range
    rng_seed = rng

    def sample(img):
        img = np.asarray(img)
        H, W, _ = _validate_img_array(img)

        rng_local = np.random.default_rng(rng_seed)
        min_r, max_r = map(int, radius_range_fixed)
        if min_r <= 0 or max_r < min_r:
            raise ValueError("radius_range must be (min_r>0, max_r>=min_r)")

        mask = np.zeros((H, W), dtype=bool)
        rr = np.arange(H)[:, None]
        cc = np.arange(W)[None, :]

        for _ in range(num_blotches_fixed):
            r0 = rng_local.integers(0, H)
            c0 = rng_local.integers(0, W)
            rad = rng_local.integers(min_r, max_r + 1)
            mask |= ((rr - r0) ** 2 + (cc - c0) ** 2) <= rad * rad

        return mask

    return sample
