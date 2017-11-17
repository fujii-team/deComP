import numpy as np
from ..utils.data import MinibatchEpochIndex
from ..utils import assertion, normalize
from .serizel import (_grad_x_l2, _grad_x_kl, _grad_d_l2, _grad_d_kl,
                      _grad_x_l2_mask, _grad_x_kl_mask,
                      _grad_d_l2_mask, _grad_d_kl_mask)


_JITTER = 1.0e-15


def solve(y, D, x, tol, minibatch, maxiter, method,
          likelihood, mask, rng, xp,
          grad_x=None, grad_d=None, alpha=1.0, beta=0.5):
    """
    Kasai, H. (2017).
    Stochastic variance reduced multiplicative update for
    nonnegative matrix factorization. Retrieved from
    https://arxiv.org/pdf/1710.10781.pdf
    """
    # mini-batch methods
    if mask is None:
        gradients_x = {'l2': _grad_x_l2,
                       'kl': _grad_x_kl,
                       'poisson': _grad_x_kl}
        gradients_d = {'l2': _grad_d_l2,
                       'kl': _grad_d_kl,
                       'poisson': _grad_d_kl}
    else:
        gradients_x = {'l2': _grad_x_l2_mask,
                       'kl': _grad_x_kl_mask,
                       'poisson': _grad_x_kl_mask}
        gradients_d = {'l2': _grad_d_l2_mask,
                       'kl': _grad_d_kl_mask,
                       'poisson': _grad_d_kl_mask}

    grad_x = gradients_x[likelihood] if grad_x is None else grad_x
    grad_d = gradients_d[likelihood] if grad_d is None else grad_d

    if method == 'svrmu':
        iter_minibatch = 1
    elif method == 'svrmu-acc':
        # TODO
        F, K = D.shape
        N = x.shape[0]
        iter_minibatch = int(np.maximum(beta * F * (3 * K + 2 * N)
                                        / (3 * F * N + 2 * K), 1.0))
    else:
        raise NotImplementedError('NMF with {} algorithm is not yet '
                                  'implemented.'.format(method))
    # with iter_minibatch
    return solve_svrmu(y, D, x, tol, minibatch, maxiter, iter_minibatch,
                       mask, rng, xp, grad_x, grad_d, alpha)


def solve_svrmu(y, D, x, tol, minibatch, maxiter, iter_minibatch,
                mask, rng, xp, grad_x, grad_d, alpha):
    """ Algorithm 1 in the paper """
    minibatch_index = MinibatchEpochIndex(len(y), minibatch, rng, xp)
    minibatch_index.shuffle()
    minibatch_num = len(minibatch_index)

    grad_D_pos_prev = xp.zeros((minibatch_num, ) + D.shape, dtype=D.dtype)
    grad_D_neg_prev = xp.zeros((minibatch_num, ) + D.shape, dtype=D.dtype)

    # first round
    grad_D_pos_full, grad_D_neg_full = grad_d(y, x, D, mask, xp)
    grad_D_pos_full /= minibatch_num
    grad_D_neg_full /= minibatch_num

    for it in range(1, maxiter):
        # compute full gradient
        grad_D_pos_full, grad_D_neg_full = grad_d(y, x, D, mask, xp)
        grad_D_pos_full /= minibatch_num
        grad_D_neg_full /= minibatch_num

        for k, index in enumerate(minibatch_index):
            x_minibatch = x[index]
            y_minibatch = y[index]
            mask_minibatch = mask[index] if mask is not None else None

            for _ in range(iter_minibatch):
                grad_x_pos, grad_x_neg = grad_x(
                            y_minibatch, x_minibatch, D, mask_minibatch, xp)
                x_minibatch = x_minibatch * xp.maximum(grad_x_pos, 0.0) \
                            / xp.maximum(grad_x_neg, _JITTER)
            x[index] = x_minibatch

            # update D
            grad_D_pos, grad_D_neg = grad_d(y_minibatch, x_minibatch, D,
                                            mask_minibatch, xp)

            Q = (grad_D_pos + grad_D_neg_prev[k]) / minibatch + grad_D_pos_full
            P = (grad_D_neg + grad_D_pos_prev[k]) / minibatch + grad_D_neg_full

            U = D - D * alpha / xp.maximum(Q, _JITTER) * (Q - P)
            D_new = normalize.l2_strict(xp.maximum(U, 0.0), axis=-1, xp=xp)
            if xp.max(xp.abs(D - D_new)) < tol:
                return it, D, x
            D = D_new

            grad_D_pos_prev[k] = grad_D_pos
            grad_D_neg_prev[k] = grad_D_neg

    return maxiter, D, x
