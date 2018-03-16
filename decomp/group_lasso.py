import numpy as np
from .utils.cp_compat import get_array_module
from .utils import assertion, dtype
from .math_utils import eigen


AVAILABLE_METHODS = ['ista']
_JITTER = 1.0e-15


def predict(A, x):
    return np.tensordot(x, A, axes=2)


def loss(y, A, alpha, x, mask=None):
    """
    Returns a loss of the grouped lasso

    Parameters
    ----------
    y: nd-array, size [..., n]
    A: 3d sparse array, size [m, g, n]
    x: nd-array, size [..., m, g]

    Returns
    -------
    loss: float
    """
    n_samples = y.shape[-1]
    if mask is not None:
        n_samples = np.sum(mask, axis=-1, keepdims=True)

    xA = np.tensordot(x, A, axes=2)
    axes = tuple(np.arange(x.ndim - 1))
    x_abs = np.sqrt(np.real(np.sum(x * np.conj(x), axis=axes, keepdims=True)))

    fit_loss = np.sum((y - xA) * np.conj(y - xA)) / (2.0 * n_samples)
    reg_loss = np.sum(alpha * x_abs)
    return fit_loss + reg_loss


def solve(y, A, alpha, x=None, method='ista', tol=1.0e-5, maxiter=1000,
          mask=None, **kwargs):
    """
    Solve a group lasso problem.

    argmin_x {1 / (2 * n) * |y - xA|^2 + alpha \sum_p \sqrt{\sum_j x^2}}

    Parameters
    ----------
    y: nd-array
        Data to be fitted, size [..., n]
    A: 3d-array
        Design matrix, sized [g, m, n]. g: group size. m: feature size.
    x: 2d-array or None.
        Initial guess of the solution, sized [..., g, m]
    alpha: float or nd-array
        Regularization parameter.
    axes: integer or a tuple of integers.
        Which axis should be grouped.
    """
    # Check all the class are numpy or cupy
    xp = get_array_module(y, A, x, mask)

    if x is None:
        x = xp.zeros(y.shape[:-1] + A.shape[:-1], dtype=y.dtype)

    assertion.assert_dtypes(y=y, A=A, x=x)
    assertion.assert_dtypes(mask=mask, dtypes='f')
    assertion.assert_nonnegative(mask)
    assertion.assert_ndim('A', A, ndim=3)
    assertion.assert_shapes('x', x, 'A', A, axes=2)
    assertion.assert_shapes('y', y, 'x', x,
                            axes=np.arange(x.ndim - 2).tolist())
    assertion.assert_shapes('y', y, 'A', A, axes=[-1])
    if mask is not None and mask.ndim == 1:
        assertion.assert_shapes('y', y, 'mask', mask, axes=[-1])
    else:
        assertion.assert_shapes('y', y, 'mask', mask)
    if method not in AVAILABLE_METHODS:
        raise ValueError('Available methods are {0:s}. Given {1:s}'.format(
                            str(AVAILABLE_METHODS), method))

    assert A.dtype.kind != 'c' or method[-4:] != '_pos'
    return solve_fastpath(y, A, alpha, x, tol, maxiter, method, xp, mask=mask,
                          **kwargs)


def solve_fastpath(y, A, alpha, x, tol, maxiter, method, xp, mask=None,
                   **kwargs):
    """ fast path for group lasso, without default value setting and
    shape/dtype assertions.

    In this method, some correction takes place,

    alpha scaling:
        We changed the model from
            argmin_x {1 / (2 * n) * |y - xA|^2 - alpha |x|}
        to
            argmin_x {1 / 2 * |y - xA|^2 - alpha |x|}
        by scaling alpha by n.
        (Make sure with mask case, n is the number of valid entries)

    A scaling
        We also scale A, so that [AAt]_i,i is 1.
    """
    positive = False
    if method[-4:] == '_pos':
        method = method[:-4]
        positive = True

    if mask is not None and mask.ndim == 1:
        y = y * mask
        A = A * mask

    # A scaling
    if A.dtype.kind != 'c':
        AAt_diag_sqrt = xp.sqrt(xp.sum(xp.square(A), axis=-1))  # size [g, m]
    else:
        AAt_diag_sqrt = xp.sqrt(xp.sum(xp.real(xp.conj(A) * A), axis=-1))
    A = A / xp.expand_dims(AAt_diag_sqrt, axis=-1)
    alpha = alpha / AAt_diag_sqrt  # size [g, m]
    tol = tol * AAt_diag_sqrt
    x = x * AAt_diag_sqrt

    if mask is None or mask.ndim == 1:
        # alpha scaling
        if mask is not None:  # mask.ndim == 1
            alpha = alpha * xp.sum(mask, axis=-1)
        else:
            alpha = alpha * A.shape[-1]
        if method == 'ista':
            it, x = _solve_ista(y, A, alpha, x, tol=tol, maxiter=maxiter,
                                xp=xp, positive=positive)
        else:
            raise NotImplementedError('Method ' + method + ' is not yet '
                                      'implemented.')
    else:
        raise NotImplementedError('Method ' + method + ' is not yet '
                                  'implemented with mask.')
    # not forget to restore x value.
    return it, x / AAt_diag_sqrt


def soft_threshold(x, y, xp):
    """
    soft-threasholding function

    x: nd array.
    y: positive float (array like)

    Returns
    -------
    if x is float
        x - y if x > y
        x + y if x < -y
        0 otherwise
    """
    axes = tuple(xp.arange(x.ndim - 1))
    x_abs = np.sqrt(np.real(xp.sum(x * xp.conj(x), axis=axes, keepdims=True)))
    sign = x / xp.maximum(x_abs, _JITTER)
    return xp.maximum(x_abs - y, 0.0) * sign


def _update(yAt, AAt, x0, Lalpha_inv, L_inv, xp):
    dx = xp.swapaxes(yAt - np.tensordot(x0, AAt, axes=2), -1, -2)
    return soft_threshold(x0 + Lalpha_inv * dx, L_inv, xp)


def _update_positive(yAt, AAt, x0, Lalpha_inv, L_inv):
    raise NotImplementedError


def _solve_ista(y, A, alpha, x, tol, maxiter, positive, xp):
    """ Fast path to solve lasso by ista method """
    updator = _update_positive if positive else _update

    At = xp.transpose(A, axes=(2, 1, 0))  # [n, g, m]
    if A.dtype.kind == 'c':
        At = xp.conj(At)
    AAt = xp.tensordot(A, At, axes=1)  # [m, g, g, m]
    AAt_flat = AAt.reshape(A.shape[0] * A.shape[1], -1)  # [m*g, g*m]
    radius = eigen.spectral_radius_Gershgorin(AAt_flat, xp, keepdims=False)
    Lalpha_inv = 1.0 / radius
    L_inv = Lalpha_inv * alpha

    yAt = np.tensordot(y, At, axes=1)  # [..., g, m]

    for i in range(maxiter):
        x_new = updator(yAt, AAt, x, Lalpha_inv, L_inv, xp)
        if np.max(xp.abs(x_new - x) - tol) < 0.0:
            return i, x_new
        x = x_new

    return maxiter - 1, x
