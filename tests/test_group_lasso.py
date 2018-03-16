import numpy as np
import pytest
from decomp import group_lasso


def construct_data(shape, n_feature, n_group, is_complex, seed=0):
    """
    Construct an example data
    """
    rng = np.random.RandomState(seed)

    A = rng.randn(n_feature, n_group, shape[-1])
    x = rng.randn(*(list(shape[:-1]) + [n_feature, n_group]))

    ind = rng.choice(np.arange(n_group), int(n_group / 2))
    x[ind] = 0.0

    if not is_complex:
        y = np.tensordot(x, A, axes=2) + rng.randn(*shape) * 0.1
        return y, A, x

    A = A + 1.0j * rng.randn(n_feature, n_group, shape[-1])
    x = x + 1.0j * rng.randn(*(list(shape[:-1]) + [n_feature, n_group]))
    x[ind] = 0.0
    y = np.tensordot(x, A, axes=2) + rng.randn(*shape) * 0.1
    return y, A, x


@pytest.mark.parametrize('shape', [[10, ], [10, 15]])
@pytest.mark.parametrize('alpha', [1.0, 0.1])
@pytest.mark.parametrize('is_complex', [False, True])
def test_decrease_loss(shape, alpha, is_complex):
    y, A, x_true = construct_data(shape, 10, 3, is_complex)

    it, x = group_lasso.solve(y, A, alpha, x=None, maxiter=1)
    loss = group_lasso.loss(y, A, alpha, x)
    for _ in range(100):
        it, x = group_lasso.solve(y, A, alpha, x=x, maxiter=10)
        new_loss = group_lasso.loss(y, A, alpha, x)

        assert new_loss <= loss + 1.0e-4
        loss = new_loss
    assert not (x == 0).all()
