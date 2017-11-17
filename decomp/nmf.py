import numpy as np
from .utils.cp_compat import get_array_module
from .utils.data import minibatch_index
from .utils import assertion, normalize
from .nmf_methods import batch_mu, serizel


BATCH_METHODS = ['mu']  # , 'spgd']
MINIBATCH_METHODS = [
    'asg-mu', 'gsg-mu', 'asag-mu', 'gsag-mu',  # Romain Serizel et al
    '',  # H. Kasai et al
    ]
_JITTER = 1.0e-15


def solve(y, D, x=None, tol=1.0e-3, minibatch=None, maxiter=1000, method='mu',
          likelihood='l2', mask=None, random_seed=None, **kwargs):
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
        One of ['l2', 'kl', 'poisson']

    mask: an array-like of Boolean (or integer, float)
        The missing point should be zero. One for otherwise.

    """

    # Check all the class are numpy or cupy
    xp = get_array_module(y, D, x)

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

    # batch methods
    if minibatch is None:
        if method == 'mu':
            return batch_mu.solve(y, D, x, tol, maxiter, likelihood, mask, xp,
                                  **kwargs)
        raise NotImplementedError('Batch-NMF with {} algorithm is not yet '
                                  'implemented.'.format(method))

    # minibatch methods
    rng = np.random.RandomState(random_seed)
    if method in ['asg-mu', 'gsg-mu', 'asag-mu', 'gsag-mu']:
        return serizel.solve(y, D, x, tol, minibatch, maxiter, method,
                             likelihood, mask, rng, xp, **kwargs)
    raise NotImplementedError('NMF with {} algorithm is not yet '
                              'implemented.'.format(method))



def multiplicative_minibatch(y, D, x, tol, maxiter, mask, minibatch,
                             minibatch_iter, rng, xp, grad_x, grad_D):
    """ NMF with minibatch update
    from
    """
    forget_rate = 1.0
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
