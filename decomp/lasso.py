import numpy as np
from .utils.cp_compat import get_array_module


def soft_threshold(x, y):
    """
    complex-value compatible soft-threasholding function.

    x: a float or complex array
    y: positive float (array like)

    Returns
    -------
    if x is float
        x - y if x > y
        x + y if x < -y
        0 otherwise

    if x is complex (amplitude: r, angle: phi)
        (r - y) * exp(1j * phi) if r > y
        0 otherwise
    """
    xp = get_array_module(x)

    if hasattr(x, 'dtype') and x.dtype.kind == 'c':
        sign = x / (xp.abs(x) + 1.0e-12 * y)
    else:
        sign = xp.sign(x)
    x = xp.abs(x) - y
    return xp.maximum(x, 0.0) * sign


def solve(y, A, alpha, x0=None, tol=1.0e-3, method='ista', maxiter=1000,
          mask=None):
    """
    Solve Lasso problem

    argmin_x {|y - xA|^2 - alpha |x|}

    with
    y: [..., n_channels]
    x: [..., n_features]
    A: [n_features, n_channels]

    Parameters
    ----------
    y: array-like (float or complex)
        Target data
    A: array-like (float or complex)
        A design matrix
    alpha: a positive float
        Regularization parameter
    x0: array-like
        An initial estimate of x
    tol: a float
        Criterion to stop iteration
    method: string 'ista' | 'fista' | 'ista_mod' | 'fista_mod'
        For ista and fista, see
            Beck, A., & Teboulle, M. (n.d.).
            A Fast Iterative Shrinkage-Thresholding Algorithm for Linear
            Inverse Problems *, 2(1), 183-202.
            http://doi.org/10.1137/080716542
        for the details.
    """
    # Check all the class are numpy or cupy
    xp = get_array_module(y, A, x0, mask)

    available_methods = ['ista', 'fista']
    if method not in available_methods:
        raise ValueError('Available methods are {0:s}. Given {1:s}'.format(
                            str(available_methods), method))

    if A.dtype.kind == 'c':
        AAt = xp.dot(A, xp.conj(A.T))
        yAt = xp.tensordot(y, xp.conj(A.T), axes=1)
    else:
        AAt = xp.dot(A, A.T)
        yAt = xp.tensordot(y, A.T, axes=1)

    L = 2.0 * xp.max(xp.abs(AAt)) / alpha

    if y.ndim == 1 and x0 is None:
        x0 = xp.zeros(A.shape[0], dtype=y.dtype)
    elif y.ndim >= 2 and x0 is None:
        x0 = xp.zeros(y.shape[:-1] + (A.shape[0], ), dtype=y.dtype)

    if method == 'ista':
        for i in range(maxiter):
            x0_new = _ista_func(yAt, AAt, x0, L, alpha, xp=xp)
            if xp.max(xp.abs(x0_new - x0)) < tol:
                return i, x0_new
            else:
                x0 = x0_new

    elif method == 'fista':
        w0 = x0
        beta = 1.0
        for i in range(maxiter):
            x0_new = _solve_ista(yAt, AAt, w0, L, alpha)
            if np.max(np.abs(x0_new - x0)) < tol:
                return i, x0_new
            else:
                beta_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * beta * beta))
                w0 = x0_new + (beta - 1.0) / beta_new * (x0_new - x0)
                x0 = x0_new
                beta = beta_new
    else:
        raise NotImplementedError
    return maxiter, x0


def _solve_ista(yAt, AAt, x0, L, alpha, xp=np):
    """
    1 iteration by ISTA method.
    This is also used as fista, where w0 is passed instead of x0
    """
    dx = yAt - xp.tensordot(x0, AAt, axes=1)
    return soft_threshold(x0 + 1.0 / (L * alpha) * dx, 1.0 / L)
