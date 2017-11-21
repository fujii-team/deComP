from ..utils.data import NoneIterator


_JITTER = 1.0e-15


def get_gradients(likelihood, mask):
    if mask is None or isinstance(mask, NoneIterator):
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

    return gradients_x, gradients_d


# --- l2 loss ---
def _grad_x_l2(y, x, d, mask, xp):
    """ Multiplicative update rule for square loss.
    Returns Positive (numerator) and negative (denominator) gradients.
    mask argument is not used but just be there to match the API .
    """
    f = xp.dot(x, d)
    return xp.dot(y, d.T), xp.dot(f, d.T)


def _grad_d_l2(y, x, d, mask, xp):
    """ update d with l2 loss """
    f = xp.dot(x, d)
    return xp.dot(x.T, y), xp.dot(x.T, f)


def _grad_x_l2_mask(y, x, d, mask, xp):
    """ update x with l2 loss with mask """
    f = xp.dot(x, d) * mask
    y = y * mask
    return xp.dot(y, d.T), xp.dot(f, d.T)


def _grad_d_l2_mask(y, x, d, mask, xp):
    """ update d with l2 loss with mask """
    f = xp.dot(x, d) * mask
    y = y * mask
    return xp.dot(x.T, y), xp.dot(x.T, f)


# --- KL loss ---
def _grad_x_kl(y, x, d, mask, xp):
    """ Multiplicative update rule for KL loss. """
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
    """ Multiplicative update rule for KL loss with mask. """
    f = xp.dot(x, d) + _JITTER
    y = y * mask
    return xp.dot(x.T, y / f), xp.dot(x.T, mask)
