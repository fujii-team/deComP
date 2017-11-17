import numpy as np
import pytest
from decomp.utils.cp_compat import numpy_or_cupy as xp
from decomp import nmf


class NMFData(object):
    """
    Test from
    Kasai, H. (2017). Stochastic variance reduced multiplicative
    update for nonnegative matrix factorization. Retrieved from
    https://arxiv.org/pdf/1710.10781.pdf
    """
    def __init__(self):
        F, N, K, b = 500, 2000, 10, 200

        self.rng = xp.random.RandomState(0)

        self.xtrue = xp.maximum(self.randn(N, K), 0.0)
        self.Dtrue = xp.maximum(self.randn(K, F), 0.0)
        self.y = xp.dot(self.xtrue, self.Dtrue)
        self.D = self.Dtrue + 0.3 * xp.maximum(
                self.randn(*self.Dtrue.shape), 0.)
        self.maxiter = 100
        v = self.rng.uniform(0.45, 1.0, size=self.y.size).reshape(self.y.shape)
        self.mask = xp.rint(v).astype(np.float32)

    def randn(self, *shape):
        return self.rng.randn(*shape).astype(np.float32)

    def test(self, method):
        it, D, x = nmf.solve(self.y, self.D, tol=1.0e-10,
                             method=method, maxiter=self.maxiter)

    def test_minibatch(self, method):
        it, D, x = nmf.solve(self.y, self.D, tol=1.0e-10,
                             method=method, maxiter=self.maxiter,
                             minibatch=200)

    def test_mask(self, method):
        it, D, x = nmf.solve(self.y, self.D, tol=1.0e-10,
                             method=method, maxiter=self.maxiter,
                             mask=self.mask)

    def test_minibatch_mask(self, method):
        it, D, x = nmf.solve(self.y, self.D, tol=1.0e-10,
                             method=method, maxiter=self.maxiter,
                             mask=self.mask, minibatch=200)


@pytest.mark.parametrize("method", nmf.BATCH_METHODS)
def test_nmf(benchmark, method):
    nmf_data = NMFData()
    it_list = benchmark(nmf_data.test, method)


@pytest.mark.parametrize("method", nmf.BATCH_METHODS)
def test_nmf_mask(benchmark, method):
    nmf_data = NMFData()
    it_list = benchmark(nmf_data.test_mask, method)


@pytest.mark.parametrize("method", nmf.MINIBATCH_METHODS)
def test_nmf_minibatch(benchmark, method):
    nmf_data = NMFData()
    it_list = benchmark(nmf_data.test_minibatch, method)


@pytest.mark.parametrize("method", nmf.MINIBATCH_METHODS)
def test_nmf_minibatch_mask(benchmark, method):
    nmf_data = NMFData()
    it_list = benchmark(nmf_data.test_minibatch_mask, method)
