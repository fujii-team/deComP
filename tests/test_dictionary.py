import unittest
from decomp.utils.cp_compat import numpy_or_cupy as xp
import decomp.dictionary_learning as dic
import decomp.utils.normalize as normalize
from .testings import allclose


class TestFloat(unittest.TestCase):
    def error(self, x, D, alpha, mask=None):
        mask = xp.ones(self.y.shape, dtype=float) if mask is None else mask
        alpha = alpha * xp.sum(mask, axis=-1, keepdims=True)
        D = normalize.l2(D, axis=-1, xp=xp)
        loss = xp.sum(0.5 / alpha * xp.square(xp.abs(
                self.y - xp.tensordot(x, D, axes=1))) * mask)
        return loss + xp.sum(xp.abs(x))

    def assert_minimum(self, x, D, alpha, tol, n=3, mask=None):
        loss = self.error(x, D, alpha, mask)
        for _ in range(n):
            dx = self.randn(*x.shape) * tol
            dD = self.randn(*D.shape) * tol
            self.assertTrue(loss < self.error(x + dx, D + dD, alpha, mask))

    def randn(self, *shape):
        return self.rng.randn(*shape)

    @property
    def method(self):
        return 'block_cd'

    def setUp(self):
        self.rng = xp.random.RandomState(0)
        self.Dtrue = self.randn(3, 5)
        xtrue = self.randn(101, 3)
        self.xtrue = xtrue * self.rng.uniform(size=303).reshape(101, 3)

        self.y = xp.dot(self.xtrue, self.Dtrue) + self.randn(101, 5) * 0.1
        self.D = self.Dtrue + self.randn(*self.Dtrue.shape) * 0.3
        self.mask = xp.rint(
                self.rng.uniform(0.45, 1, size=505)).reshape(101, 5)

    def test_run(self):
        D = self.D.copy()
        alpha = 0.01
        maxiter = 1000
        it, D, x = dic.solve(self.y, D, alpha, x=None, tol=1.0e-4,
                             method=self.method,
                             minibatch=100, maxiter=maxiter,
                             lasso_method='ista', lasso_iter=1000,
                             random_seed=0)
        self.assertTrue(it < maxiter - 1)
        self.assert_minimum(x, D, alpha, tol=1.0e-3, n=3)
        self.assertFalse(allclose(x, xp.zeros_like(x)))

    """
    def test_run_minibatch_mask(self):
        D = self.D.copy()
        alpha = 0.01
        maxiter = 1000
        y = self.mask * self.y
        it, D, x = dic.solve(y, D, alpha, x=None, tol=1.0e-3,
                             minibatch=1, maxiter=maxiter,
                             lasso_method='ista', lasso_iter=100,
                             random_seed=0, mask=self.mask)
        self.assertTrue(it < maxiter - 1)
        self.assertFalse(allclose(x, xp.zeros_like(x)))
        D = self.D.copy()
        self.assert_minimum(x, D, alpha, tol=1.0e-3, n=3, mask=self.mask)
        # make sure that the solution is different from
        it2, D2, x2 = dic.solve(self.y, D, alpha, x=None, tol=1.0e-5,
                                minibatch=10, maxiter=maxiter,
                                lasso_method='ista', lasso_iter=10,
                                random_seed=0)
        self.assertFalse(allclose(D, D2, atol=1.0e-4))
    """


class TestComplex(TestFloat):
    def randn(self, *shape):
        return self.rng.randn(*shape) + self.rng.randn(*shape) * 1.0j

    def error(self, x, D, alpha, mask=None):
        mask = xp.ones(self.y.shape, dtype=float) if mask is None else mask
        alpha = alpha * xp.sum(mask, axis=-1, keepdims=True)
        D = D / xp.maximum(xp.real(xp.sum(xp.conj(D) * D, axis=0)), 1.0)
        loss = xp.sum(0.5 / alpha * xp.square(xp.abs(
                self.y - xp.tensordot(x, D, axes=1))) * mask)
        return loss + xp.sum(xp.abs(x))


class Test_method_equivalence(unittest.TestCase):
    def randn(self, *shape):
        return self.rng.randn(*shape)

    def setUp(self):
        self.rng = xp.random.RandomState(0)
        self.Dtrue = self.randn(3, 5)
        xtrue = self.randn(101, 3)
        self.xtrue = xtrue * self.rng.uniform(size=303).reshape(101, 3)

        self.y = xp.dot(self.xtrue, self.Dtrue) + self.randn(101, 5) * 0.1
        self.D = self.Dtrue + self.randn(*self.Dtrue.shape) * 0.3
        self.mask = xp.rint(
                self.rng.uniform(0.4, 1, size=505)).reshape(101, 5)
        self.alpha = 0.01
        self.methods = ['block_cd']

"""
    def test_compare(self):
        it_base, D_base, _ = dic.solve(
                self.y, self.D.copy(), self.alpha, x=None, tol=1.0e-5,
                minibatch=None, method='block_cd_fullbatch',
                maxiter=10000, lasso_method='ista', lasso_iter=10,
                random_seed=0)
        for method in self.methods:
            it, D, x = dic.solve(
                    self.y, self.D.copy(), self.alpha, x=None, tol=1.0e-5,
                    minibatch=None, method=method,
                    maxiter=10000, lasso_method='ista', lasso_iter=10,
                    random_seed=0)
            self.assertTrue(it < 10000 - 1)
            self.assertTrue(allclose(D_base, D, atol=1.0e-4))

    def test_compare_mask(self):
        it_base, D_base, _ = dic.solve(
                self.y, self.D.copy(), self.alpha, x=None, tol=1.0e-5,
                minibatch=10, method='parallel_cd',
                maxiter=10000, lasso_method='ista', lasso_iter=10,
                random_seed=0, mask=self.mask)
        for method in self.methods:
            it, D, x = dic.solve(
                    self.y, self.D.copy(), self.alpha, x=None, tol=1.0e-5,
                    minibatch=10, method=method,
                    maxiter=10000, lasso_method='ista', lasso_iter=10,
                    random_seed=0, mask=self.mask)
            self.assertTrue(it < 10000 - 1)
            self.assertTrue(allclose(D_base, D, atol=1.0e-4))
"""


if __name__ == '__main__':
    unittest.main()
