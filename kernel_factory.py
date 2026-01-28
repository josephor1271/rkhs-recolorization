import numpy as np

############################# Main Functions #############################
##########################################################################


def make_nonlocal_kernel_factory(*, t: float, p: float = 2.0, eps: float = 1e-12):
    """
    Returns a function make_kernel(g) that builds the nonlocal kernel:

      k(x,y) = exp( - |g(x) - g(y)|^p / (4t) )

    where g: [0,1]^2 -> R is the grayscale function.
    """
    t = float(t)
    p = float(p)
    eps = float(eps)

    if t <= 0:
        raise ValueError(f"t must be > 0, got {t}")
    if not (0 < p <= 2):
        raise ValueError(f"p must be in (0,2], got {p}")

    denom = 4.0 * t + eps

    def make_kernel(g):
        def k(x, y) -> float:
            gx = float(g(x))
            gy = float(g(y))
            diff = abs(gx - gy)
            return float(np.exp(-(diff**p) / denom))

        return k

    return make_kernel


def make_local_kernel_factory(*, t: float, p: float = 2.0, eps: float = 1e-12):
    """
    Returns a function make_kernel(g) that builds the local/spatial kernel:

      k_local(x,y) = exp( - ||x - y||^p / (4t) )

    Note: g is accepted for API consistency but not used.
    """
    t = float(t)
    p = float(p)
    eps = float(eps)

    if t <= 0:
        raise ValueError(f"t must be > 0, got {t}")
    if not (0 < p <= 2):
        raise ValueError(f"p must be in (0,2], got {p}")

    denom = 4.0 * t + eps

    def make_kernel(_):
        def k_local(x, y) -> float:
            x1, x2 = x
            y1, y2 = y
            dx = float(x1) - float(y1)
            dy = float(x2) - float(y2)
            dist = np.sqrt(dx * dx + dy * dy)
            return float(np.exp(-(dist**p) / denom))

        return k_local

    return make_kernel


def combine_kernel_factories(kf1, kf2, sigma1, sigma2):
    """
    Combine two kernel factories by multiplication, with constant weights:
      k(x,y) = (exp(-2*sigma1) * k1(x,y)) * (exp(-2*sigma2) * k2(x,y))
    """
    w1 = float(np.exp(-2.0 * float(sigma1)))
    w2 = float(np.exp(-float(sigma2)))

    def make_kernel(g):
        k1 = kf1(g)
        k2 = kf2(g)

        def k(x, y) -> float:
            return float((w1 * k1(x, y)) * (w2 * k2(x, y)))

        return k

    return make_kernel



