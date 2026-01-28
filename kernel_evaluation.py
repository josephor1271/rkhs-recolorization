import numpy as np

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

def mask_to_D(
    mask: np.ndarray, *, ordering="row-major", rng=None, as_set: bool = False
):
    """
    Convert a (H,W) boolean mask into normalized coordinates in [0,1)^2.

    Convention (matches your examples):
      - pixel (row=0,col=0) -> (0.0, 0.0)
      - pixel (row=H-1,col=W-1) -> ((W-1)/W, (H-1)/H)
        e.g. 100x100 -> (0.99, 0.99)

    Returns:
      - If as_set=False: np.ndarray of shape (m,2) with rows [x1, x2] = [col/W, row/H]
      - If as_set=True:  set of (x1,x2) tuples (order lost)
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (H,W), got {mask.shape}")

    H, W = mask.shape
    rows, cols = np.nonzero(mask)  # already row-major sorted by row, then col

    if ordering == "col-major":
        order = np.lexsort((rows, cols))  # sort by col then row
        rows, cols = rows[order], cols[order]
    elif ordering == "random":
        rng = np.random.default_rng(rng)
        perm = rng.permutation(rows.size)
        rows, cols = rows[perm], cols[perm]
    elif ordering != "row-major":
        raise ValueError("ordering must be 'row-major', 'col-major', or 'random'")

    x1 = cols.astype(np.float32) / float(W)  # col/W
    x2 = rows.astype(np.float32) / float(H)  # row/H

    coords = np.stack([x1, x2], axis=1)  # (m,2)

    if as_set:
        return set(map(tuple, coords.tolist()))
    return coords


def omega_coords(H: int, W: int, *, dtype=np.float32) -> np.ndarray:
    """
    Return all Ω coordinates for an HxW image in row-major order.
    Bin convention:
      (row=r, col=c) -> (x1, x2) = (c/W, r/H)
    Output shape: (H*W, 2) with rows [x1, x2].
    """
    H = int(H)
    W = int(W)
    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive")

    rr, cc = np.indices((H, W))
    x1 = (cc.astype(dtype) / W).ravel()  # col/W
    x2 = (rr.astype(dtype) / H).ravel()  # row/H
    return np.stack([x1, x2], axis=1)


def kernel_gram_matrix(k, D: np.ndarray, *, dtype=np.float64) -> np.ndarray:
    """
    Compute K_D where (K_D)[i,j] = k(D[i], D[j]) for D ⊂ Ω.

    Args:
      k: scalar kernel function k(x,y) -> float
      D: (m,2) array of normalized coordinates in [0,1)^2
      symmetric: if True, fill only upper triangle and mirror (assumes k(x,y)=k(y,x))
      dtype: dtype of returned matrix

    Returns:
      K_D: (m,m) numpy array
    """
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[1] != 2:
        raise ValueError(f"D must have shape (m,2), got {D.shape}")

    m = D.shape[0]
    K = np.empty((m, m), dtype=dtype)
    print(f"K_D matrix is {m} x {m} size, {m*m} entries")

    for i in range(m):
        xi = (float(D[i, 0]), float(D[i, 1]))
        K[i, i] = k(xi, xi)
        for j in range(i + 1, m):
            xj = (float(D[j, 0]), float(D[j, 1]))
            kij = k(xi, xj)
            K[i, j] = kij
            K[j, i] = kij

    return K


def kernel_cross_matrix_Omega_D(
    k,
    D: np.ndarray,
    H: int,
    W: int,
    *,
    dtype=np.float64,
) -> np.ndarray:
    """
    Compute K_cD of shape (N, m) where N=H*W and m=len(D).
    Row index u corresponds to pixel (row=r, col=c) in row-major order:
      u = r*W + c
    and u's normalized coord is (c/W, r/H) (bin convention).

    Args:
      k: scalar kernel k(x,y) -> float
      D: (m,2) array of normalized coordinates (same convention)
      H, W: image height/width
      dtype: dtype for output matrix

    Returns:
      K_cD: (H*W, m) numpy array
    """
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[1] != 2:
        raise ValueError(f"D must have shape (m,2), got {D.shape}")

    H = int(H)
    W = int(W)
    if H <= 0 or W <= 0:
        raise ValueError("H and W must be positive")

    m = D.shape[0]
    N = H * W
    KcD = np.empty((N, m), dtype=dtype)
    print(f"K_cD matrix is {N} x {m} size, {N*m} entries")

    # Pre-pack D points as tuples for faster inner loop
    D_tuples = [(float(D[i, 0]), float(D[i, 1])) for i in range(m)]

    idx = 0
    for r in range(H):
        x2 = r / H
        for c in range(W):
            x1 = c / W
            u = (x1, x2)
            # fill row idx
            for j, xj in enumerate(D_tuples):
                KcD[idx, j] = k(u, xj)
            idx += 1

    return KcD


def mask_to_rc(
    mask: np.ndarray,
    *,
    ordering="row-major",
    rng=None,
) -> np.ndarray:
    """
    Return integer (row,col) coordinates of True entries in mask.
    Output shape: (m,2) with rows [row, col]
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (H,W), got {mask.shape}")

    rows, cols = np.nonzero(mask)  # row-major order by default
    if ordering == "col-major":
        order = np.lexsort((rows, cols))  # by col then row
        rows, cols = rows[order], cols[order]
    elif ordering == "random":
        rng = np.random.default_rng(rng)
        perm = rng.permutation(rows.size)
        rows, cols = rows[perm], cols[perm]
    elif ordering != "row-major":
        raise ValueError("ordering must be 'row-major', 'col-major', or 'random'")

    return np.stack([rows, cols], axis=1)


def rc_to_D(rc: np.ndarray, H: int, W: int, *, dtype=np.float32) -> np.ndarray:
    """
    Convert integer (row,col) -> normalized (x1,x2) in bin convention:
      x1 = col/W, x2 = row/H
    """
    rc = np.asarray(rc)
    rows = rc[:, 0].astype(dtype)
    cols = rc[:, 1].astype(dtype)
    x1 = cols / float(W)
    x2 = rows / float(H)
    return np.stack([x1, x2], axis=1)


def sample_rgb_on_mask(
    color_array: np.ndarray,
    mask: np.ndarray,
    *,
    ordering="row-major",
    rng=None,
    dtype=np.float64,
):
    """
    Returns:
      D:   (m,2) normalized coords (bin convention)
      F_D: (m,3) RGB values at those coords (same order)
      rc:  (m,2) integer row/col (same order)
    """
    color_array = np.asarray(color_array)
    mask = np.asarray(mask, dtype=bool)

    if color_array.ndim != 3 or color_array.shape[2] != 3:
        raise ValueError(f"color_array must be (H,W,3), got {color_array.shape}")

    H, W, _ = color_array.shape
    if mask.shape != (H, W):
        raise ValueError(f"mask must be shape {(H,W)}, got {mask.shape}")

    rc = mask_to_rc(mask, ordering=ordering, rng=rng)  # (m,2)
    rows, cols = rc[:, 0], rc[:, 1]

    F_D = color_array[rows, cols, :].astype(dtype, copy=False)  # (m,3)
    D = rc_to_D(rc, H, W)

    return D, F_D, rc


def solve_for_A(
    K_D: np.ndarray,
    F_D: np.ndarray,
    *,
    gamma: float = 1e-4,
    jitter: float = 1e-10,
) -> np.ndarray:
    """
    Solve (K_D + gamma*m*I) A = F_D for A, where:
      K_D: (m,m) kernel Gram matrix on D
      F_D: (m,3) known RGB values at points in D (ordered consistently with K_D)
      A:   (m,3) coefficients (one column per channel)

    Args:
      gamma: regularization strength (small positive)
      jitter: extra diagonal stabilizer

    Returns:
      A: (m,3)
    """
    K_D = np.asarray(K_D, dtype=np.float64)
    F_D = np.asarray(F_D, dtype=np.float64)

    if K_D.ndim != 2 or K_D.shape[0] != K_D.shape[1]:
        raise ValueError(f"K_D must be square (m,m), got {K_D.shape}")
    m = K_D.shape[0]

    if F_D.shape != (m, 3):
        raise ValueError(f"F_D must have shape (m,3) with m={m}, got {F_D.shape}")

    gamma = float(gamma)
    jitter = float(jitter)
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")
    if jitter < 0:
        raise ValueError(f"jitter must be >= 0, got {jitter}")

    M = K_D + (gamma * m + jitter) * np.eye(m)

    L = np.linalg.cholesky(M)
    # Solve L Y = F_D, then L^T A = Y
    Y = np.linalg.solve(L, F_D)
    A = np.linalg.solve(L.T, Y)
    return A


def reconstruct_image_via_KcD(
    K_cD: np.ndarray, A: np.ndarray, H: int, W: int, *, clip: bool = True
) -> np.ndarray:
    """
    Given K_cD (H*W, m) and A (m,3), compute F_img (H,W,3).
    Keeps values in the same scale as A (normalized if your training data was normalized).
    """
    K_cD = np.asarray(K_cD, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)

    H = int(H)
    W = int(W)
    N = H * W

    if K_cD.ndim != 2:
        raise ValueError(f"K_cD must be 2D, got {K_cD.shape}")
    if K_cD.shape[0] != N:
        raise ValueError(f"K_cD must have {N} rows (H*W), got {K_cD.shape[0]}")
    m = K_cD.shape[1]
    if A.shape != (m, 3):
        raise ValueError(f"A must have shape (m,3) with m={m}, got {A.shape}")

    F_flat = K_cD @ A  # (H*W, 3)
    F_img = F_flat.reshape(H, W, 3)

    if clip:
        F_img = np.clip(F_img, 0.0, 1.0)

    return F_img