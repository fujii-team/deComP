import numpy as np
from .utils.cp_compat import get_array_module
from .utils.data import minibatch_index
from .utils import assertion, normalize


AVAILABLE_METHODS = ['multiplicative', 'spgd']
_JITTER = 1.0e-15


def solve(y, D, x=None, tol=1.0e-3,
          minibatch=None, maxiter=1000, method='multiplicative',
          likelihood='l2', minibatch_iter=100, mask=None,
          random_seed=None):
    """
    Non-negative matrix factrization.

    argmin_{x, D} {|y - xD|^2 - alpha |x|}
    s.t. |D_j|^2 <= 1 and D > 0 and x > 0

    with
    y: [n_samples, n_channels]
    x: [n_samples, n_features]
    D: [n_features, n_channels]

    Parameters
    ----------
    y: array-like.
        Shape: [n_samples, ch]
    D: array-like.
        Initial dictionary, shape [ch, n_component]
    x0: array-like
        An initial estimate of x
    tol: a float.
        Criterion to stop iteration
    maxiter: an integer
        Number of iteration
    method: string
        One of AVAILABLE_METHODS
    likelihood: string
        One of ['l2']

    mask: an array-like of Boolean (or integer, float)
        The missing point should be zero. One for otherwise.

    """

    # Check all the class are numpy or cupy
    xp = get_array_module(y, D, x)

    rng = np.random.RandomState(random_seed)
    if x is None:
        x = xp.ones((y.shape[0], D.shape[0]), dtype=y.dtype)

    assertion.assert_dtypes(y=y, D=D, x=x)
    assertion.assert_dtypes(y=y, D=D, x=x, mask=mask, dtypes='f')
    assertion.assert_shapes('x', x, 'D', D, axes=1)
    assertion.assert_shapes('y', y, 'D', D, axes=[-1])
    assertion.assert_shapes('y', y, 'mask', mask)
    assertion.assert_ndim('y', y, 2)
    assertion.assert_ndim('D', D, 2)
    assertion.assert_ndim('x', x, 2)
    assertion.assert_nonnegative(D)
    assertion.assert_nonnegative(x)

    if likelihood not in ['l2']:
        assertion.assert_nonnegative(y)

    D = normalize.l2_strict(D, axis=-1, xp=xp)
    if method == 'multiplicative':
        if mask is None:
            gradients_x = {'l2': _grad_x_l2,
                           'kl': _grad_x_kl,
                           'poisson', _grad_x_kl}
            gradients_d = {'l2': _grad_d_l2,
                           'kl': _grad_d_kl,
                           'poisson', _grad_d_kl}
        else:
            gradients_x = {'l2': _grad_x_l2_mask,
                           'kl': _grad_x_kl_mask,
                           'poisson', _grad_x_kl_mask}
            gradients_d = {'l2': _grad_d_l2_mask,
                           'kl': _grad_d_kl_mask,
                           'poisson', _grad_d_kl_mask}

        if minibatch is None:
            return multiplicative_fullbatch(y, D, x, tol, maxiter, mask, xp,
                                            grad_x=gradients_x[likelihood],
                                            grad_D=gradients_d[likelihood])
        else:
            raise NotImplementedError('NMF with minibatch is not yet '
                                      'implemented.')
            return multiplicative_minibatch(
                y, D, x, tol, maxiter, mask, minibatch, minibatch_iter,
                rng, xp,
                grad_x=gradients_x[likelihood],
                grad_D=gradients_d[likelihood])
    else:
        raise NotImplementedError('NMF with {} algorithm is not yet '
                                  'implemented.'.format(method))


# --- l2 loss ---
def _grad_x_l2(y, x, d, mask, xp):
    """ Multiplicative update rule for square loss.
    Returns Positive (numerator) and negative (denominator).
    mask is not used.
    """
    f = xp.dot(x, d)
    return xp.dot(y, d.T), xp.dot(f, d.T)


def _grad_d_l2(y, x, d, mask, xp):
    """ update d with l2 loss """
    f = xp.dot(x, d)
    return xp.dot(x.T, y), xp.dot(x.T, f)


def _grad_x_l2_mask(y, x, d, mask, xp):
    """ Multiplicative update rule for square loss.
    Returns Positive (numerator) and negative (denominator).
    mask is not used.
    """
    f = xp.dot(x, d) * mask
    y = y * mask
    return xp.dot(y, d.T), xp.dot(f, d.T)


def _grad_d_l2_mask(y, x, d, mask, xp):
    """ update d with l2 loss """
    f = xp.dot(x, d) * mask
    y = y * mask
    return xp.dot(x.T, y), xp.dot(x.T, f)


# --- KL loss ---
def _grad_x_kl(y, x, d, mask, xp):
    """ Multiplicative update rule for KL loss.    """
    f = xp.dot(x, d) + _JITTER
    return xp.dot(y / f, d.T), xp.sum(d.T, axis=0, keepdims=True)


def _grad_d_kl(y, x, d, mask, xp):
    """ update d with KL loss """
    f = xp.dot(x, d) + _JITTER
    return xp.dot(x.T, y / f), xp.sum(x.T, axis=1, keepdims=True)


def _grad_x_kl_mask(y, x, d, mask, xp):
    """ Multiplicative update rule for KL loss with mask. """
    f = xp.dot(x, d) + _JITTER
    y = y * mask
    return xp.dot(y / f, d.T), xp.dot(mask, d.T)


def _grad_d_kl_mask(y, x, d, mask, xp):
    """ update d with l2 loss """
    f = xp.dot(x, d) + _JITTER
    y = y * mask
    return xp.dot(x.T, y / f), xp.dot(x.T, mask)


def multiplicative_fullbatch(y, D, x, tol, maxiter, mask, xp, grad_x, grad_D):
    """ NMF with fullbatch update """
    for it in range(1, maxiter):
        # update x
        grad_x_pos, grad_x_neg = grad_x(y, x, D, mask, xp)
        x = x * xp.maximum(grad_x_pos, 0.0) / xp.maximum(grad_x_neg, _JITTER)
        # update D
        grad_D_pos, grad_D_neg = grad_D(y, x, D, mask, xp)
        U = D * xp.maximum(grad_D_pos, 0.0) / xp.maximum(grad_D_neg, _JITTER)
        D_new = normalize.l2_strict(U, axis=-1, xp=xp)
        if it % 100 == 0:
            print(xp.max(xp.abs(D - D_new)))

        if xp.max(xp.abs(D - D_new)) < tol:
            return it, D_new, x
        D = D_new

    return maxiter, D, x


def multiplicative_minibatch(y, D, x, tol, maxiter, mask, minibatch,
                             minibatch_iter, rng, xp, grad_x, grad_D):
    """ NMF with minibatch update
    from
    """
    forget_rate = 0.1
    grad_D_pos_sum = xp.zeros_like(D)
    grad_D_neg_sum = xp.zeros_like(D)

    def accumurate_grad(grad_sum, grad):
        return (1.0 - forget_rate) * grad_sum + forget_rate * grad

    for it in range(1, maxiter):
        indexes = minibatch_index(y.shape[:1], minibatch, rng)
        x_minibatch = x[indexes]
        y_minibatch = y[indexes]
        mask_minibatch = mask[indexes] if mask is not None else None
        # update x
        for minibatch_it in range(minibatch_iter):
            grad_x_pos, grad_x_neg = grad_x(
                            y_minibatch, x_minibatch, D, mask_minibatch, xp)
            x_minibatch = x_minibatch * \
                xp.maximum(grad_x_pos, 0.0) / xp.maximum(grad_x_neg, _JITTER)
        x[indexes] = x_minibatch

        # update D
        grad_D_pos, grad_D_neg = grad_D(y_minibatch, x_minibatch, D,
                                        mask_minibatch, xp)
        grad_D_pos_sum = accumurate_grad(grad_D_pos_sum, grad_D_pos)
        grad_D_neg_sum = accumurate_grad(grad_D_neg_sum, grad_D_neg)

        U = D * xp.maximum(grad_D_pos_sum, 0.0) / xp.maximum(grad_D_neg_sum,
                                                             _JITTER)
        D_new = normalize.l2_strict(U, axis=-1, xp=xp)
        if xp.max(xp.abs(D - D_new)) < tol:
            return it, D, x
        D = D_new

        if it % 10 == 0:
            #print(grad_D_pos_sum[0, :4])
            print(xp.sum(xp.square(y - xp.dot(x, D))))
    return maxiter, D, x
