""" Collection of eigen value estimator """


def spectral_radius_svd(X, xp):
    """ Largest magnitude of eigen values by SVD """
    return xp.linalg.svd(X)[1][0]


def spectral_radius_Gershgorin(X, xp, keepdims=False):
    """
    An upper bound for the largest eigen velue by
    Gershgorin circle theorem.
    https://en.wikipedia.org/wiki/Gershgorin_circle_theorem

    Here, we assume X is a symmetric matrix or a stack of matrices.
    X.shape: [..., n, n]

    Return:
    -------
    [..., 1, 1] if keepdims is True
    [...,] if keepdims is False
    """
    #if keepdims:
    #    return xp.max(xp.sum(xp.abs(X), axis=-2, keepdims=True),
    #                  axis=-2, keepdims=keepdims)
    return xp.max(xp.sum(xp.abs(X), axis=-2, keepdims=keepdims),
                  axis=-1, keepdims=keepdims)
