import unittest
from decomp.utils.cp_compat import numpy_or_cupy as xp
from decomp.utils import normalize
import decomp.nmf as nmf
from .testings import allclose


class Test_L2(unittest.TestCase):
    def randn(self, *shape):
        return self.rng.randn(*shape)

    @property
    def likelihood(self):
        return 'l2'

    def get_y(self, x, d):
        ytrue = xp.dot(x, d)
        return ytrue + self.randn(*ytrue.shape) * 0.1

    def setUp(self):
        self.rng = xp.random.RandomState(0)
        self.Dtrue = xp.maximum(self.randn(3, 20), 0.0)
        self.xtrue = xp.maximum(self.randn(101, 3), 0.0)

        self.y = self.get_y(self.xtrue, self.Dtrue)
        self.D = xp.maximum(self.Dtrue + self.randn(*self.Dtrue.shape) * 0.3,
                            0.1)
        self.mask = xp.rint(
            self.rng.uniform(0.3, 1, size=self.y.size)).reshape(self.y.shape)

    def error(self, x, D, mask):
        mask = xp.ones(self.y.shape, self.y.dtype) if mask is None else mask
        D = normalize.l2_strict(D, axis=-1, xp=xp)
        loss = xp.sum(xp.square(self.y - xp.dot(x, D)) * mask)
        return 0.5 * loss

    def assert_minimum(self, x, D, tol, n=100, mask=None):
        loss = self.error(x, D, mask)
        for _ in range(n):
            dx = self.randn(*x.shape) * tol
            dD = self.randn(*D.shape) * tol
            assert loss < self.error(xp.maximum(x + dx, 0.0),
                                     xp.maximum(D + dD, 0.0), mask) + 1.0e-15

    @property
    def maxiter(self):
        return 3000

    @property
    def minibatch_maxiter(self):
        return 1000

    def test_run(self):
        D = self.D.copy()
        it, D, x = nmf.solve(self.y, D, x=None, tol=1.0e-6,
                             minibatch=None, maxiter=self.maxiter,
                             method='multiplicative',
                             likelihood=self.likelihood, mask=None,
                             random_seed=0)
        assert it < self.maxiter - 1
        self.assert_minimum(x, D, tol=1.0e-5, n=100)
        assert not allclose(D, xp.zeros_like(D), atol=1.0e-5)

    def test_run_mask(self):
        D = self.D.copy()
        it, D, x = nmf.solve(self.y, D, x=None, tol=1.0e-6,
                             minibatch=None, maxiter=self.maxiter,
                             method='multiplicative',
                             likelihood=self.likelihood, mask=self.mask,
                             random_seed=0)
        assert it < self.maxiter - 1
        self.assert_minimum(x, D, tol=1.0e-5, n=100, mask=self.mask)
        assert not allclose(D, xp.zeros_like(D), atol=1.0e-5)

    def _test_run_minibatch(self):
        D_minibatch = self.D.copy()
        it, D_minibatch, x_minibatch = nmf.solve(
                     self.y, D_minibatch, x=None, tol=1.0e-3,
                     minibatch=10, maxiter=self.minibatch_maxiter,
                     minibatch_iter=1,
                     method='multiplicative',
                     likelihood=self.likelihood, mask=None,
                     random_seed=0)
        assert it < self.minibatch_maxiter - 1
        print(D)
        print(D - D_minibatch)
        assert allclose(D, D_minibatch, atol=1.0e-2)

    '''
    def test_run_minibatch_mask(self):
        D = self.D.copy()
        y = self.mask * self.y
        it, D, x = nmf.solve(self.y, D, minibatch=10, maxiter=1000,
                             tol=1.0e-6, mask=self.mask, random_seed=0)
        self.assertTrue(it < 1000 - 1)
        self.assert_minimum(x, D, tol=1.0e-5, n=3)
        # make sure that the solution is different from
        D = self.D.copy()
        it2, D2, x2 = nmf.solve(self.y, D, minibatch=10, maxiter=1000,
                                tol=1.0e-6, mask=None, random_seed=0)
        self.assertFalse(allclose(D, D2, atol=1.0e-4))
    '''


class Test_KL(Test_L2):
    @property
    def likelihood(self):
        return 'kl'

    def get_y(self, x, d):
        ytrue = xp.dot(x, d)
        return ytrue + xp.abs(self.randn(*ytrue.shape) * 0.1)

    def error(self, x, D, mask):
        mask = xp.ones(self.y.shape, self.y.dtype) if mask is None else mask
        D = normalize.l2_strict(D, axis=-1, xp=xp)
        f = xp.maximum(xp.dot(x, D), 1.0e-15)
        return xp.sum((self.y * xp.log(xp.maximum(self.y, 1.0e-15) / f)
                      + f) * mask)


if __name__ == '__main__':
    unittest.main()
