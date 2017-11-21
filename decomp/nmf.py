import numpy as np
from .utils.cp_compat import get_array_module
from .utils.data import (minibatch_index, MinibatchData, NoneIterator,
                         SequentialMinibatchData, ParallelMinibatchData)
from .utils import assertion, normalize
from .nmf_methods import batch_mu, serizel, kasai


BATCH_METHODS = ['mu']  # , 'spgd']
MINIBATCH_METHODS = [
    'asg-mu', 'gsg-mu', 'asag-mu', 'gsag-mu',  # Romain Serizel et al
    'svrmu', 'svrmu-acc',  # H. Kasai et al
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
    xp = get_array_module(D)
    get_array_module(y, x)  # make sure y and x is the same type
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
        # Check all the class are numpy or cupy
        xp = get_array_module(y, D, x)
        if method == 'mu':
            return batch_mu.solve(y, D, x, tol, maxiter, likelihood, mask, xp,
                                  **kwargs)
        raise NotImplementedError('Batch-NMF with {} algorithm is not yet '
                                  'implemented.'.format(method))

    # minibatch methods
    # prepare MinibatchData
    if xp is np or get_array_module(y) is not np: # both np or xp
        # just check the array type
        xp = get_array_module(y, D, x, mask)
        y = MinibatchData(y, minibatch)
        x = MinibatchData(x, minibatch)
        if mask is None:
            mask = NoneIterator()
        else:
            mask = MinibatchData(mask, minibatch)
        rng = xp.random.RandomState(random_seed)
    else:  # np.store and cupy calculation
        y = SequentialMinibatchData(y, minibatch, needs_update=False)
        x = SequentialMinibatchData(x, minibatch)
        if mask is None:
            mask = NoneIterator()
        else:
            mask = SequentialMinibatchData(mask, minibatch, needs_update=False)
        rng = np.random.RandomState(random_seed)

    if method in ['asg-mu', 'gsg-mu', 'asag-mu', 'gsag-mu']:
        return serizel.solve(y, D, x, tol, minibatch, maxiter, method,
                             likelihood, mask, rng, xp, **kwargs)
    if method in ['svrmu', 'svrmu-acc']:
        return kasai.solve(y, D, x, tol, minibatch, maxiter, method,
                           likelihood, mask, rng, xp, **kwargs)

    raise NotImplementedError('NMF with {} algorithm is not yet '
                              'implemented.'.format(method))
