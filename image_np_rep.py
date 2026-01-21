import numpy as np
from PIL import Image

def rgb_image_to_array(im:Image.Image, dtype=np.uint8) -> np.ndarray:
    """
    Input:  PIL Image (any mode)
    Output: (H, W, 3) numpy array
    """
    if im.mode != "RGB":
        im = im.convert("RGB")
    arr = np.asarray(im, dtype=dtype)
    # arr shape should be (H, W, 3)
    return arr


def rgba_image_to_array(im:Image.Image, dtype=np.uint8) -> np.ndarray:
    """
    Input:  PIL Image (any mode)
    Output: (H, W, 4) numpy array (RGBA)
    """
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    arr = np.asarray(im, dtype=dtype)
    # arr shape should be (H, W, 4)
    return arr


def gray_image_to_array(im:Image.Image, dtype=np.uint8) -> np.ndarray:
    """
    Input:  PIL Image (any mode)
    Output: (H, W) numpy array (grayscale, mode 'L')
    """
    if im.mode != "L":
        im = im.convert("L")
    arr = np.asarray(im, dtype=dtype)
    # arr shape should be (H, W)
    return arr


def normalize_image(arr: np.ndarray) -> np.ndarray:
    """
    Convert uint8-like arrays to float32 in [0,1].
    Works for (H,W), (H,W,3), (H,W,4).
    """
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        # Assume already float; optionally you could clip here
        return arr.astype(np.float32, copy=False)
    return (arr.astype(np.float32) / 255.0)


def revert_image(arr: np.ndarray) -> np.ndarray:
    """
    Convert float array (expected [0,1]) to uint8 safely.
    """
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)