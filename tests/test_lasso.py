import unittest
import numpy as np
from decomp.utils.cp_compat import numpy_or_cupy as xp
from decomp import lasso
from .testings import allclose


class TestUtils(unittest.TestCase):
    def setUp(self):
        pass

    def test_soft_threshold(self):
        # test with scalar
        actual = lasso.soft_threshold(0.1, 1.0)
        expected = 0.0
        self.assertTrue(allclose(actual, expected))

        # test with 1d-array
        actual = lasso.soft_threshold([0.1, -2.0, 1.4], 1.0)
        expected = [0.0, -1.0, 0.4]
        self.assertTrue(allclose(actual, expected))

        # test with 2d-array
        actual = lasso.soft_threshold([[0.1, -2.0, 1.4],
                                       [1.1, 3.0, -1.4]], 1.0)
        expected = [[0.0, -1.0, 0.4],
                    [0.1, 2.0, -0.4]]
        self.assertTrue(allclose(actual, expected))

    def test_soft_threshold_complex(self):
        # test with 1d-array
        # real
        x = xp.array([0.1, -2.0, 1.4])
        actual = lasso.soft_threshold(x + 0.0j, 1.0)
        expected = lasso.soft_threshold(x, 1.0)
        self.assertTrue(allclose(actual, expected))
        # imag
        actual = lasso.soft_threshold(x * 1.0j, 1.0)
        expected = lasso.soft_threshold(x, 1.0) * 1.0j
        self.assertTrue(allclose(actual, expected))

        # 45deg
        actual = lasso.soft_threshold(x + x * 1.0j, 1.0)
        expected = lasso.soft_threshold(x * np.sqrt(2.0), 1.0)\
            * (np.sqrt(0.5) + np.sqrt(0.5) * 1.0j)
        self.assertTrue(allclose(actual, expected))
        # -45deg
        actual = lasso.soft_threshold(x - x * 1.0j, 1.0)
        expected = lasso.soft_threshold(x * np.sqrt(2.0), 1.0)\
            * (np.sqrt(0.5) - np.sqrt(0.5) * 1.0j)
        self.assertTrue(allclose(actual, expected))


class TestLasso(unittest.TestCase):
    def randn(self, *shape):
        return self.rng.randn(*shape)

    def setUp(self):
        self.rng = xp.random.RandomState(0)
        self.A = self.randn(5, 10)
        self.x_true = self.randn(5) * xp.rint(self.rng.uniform(size=5))
        self.y = xp.dot(self.x_true,
                        self.A) + self.randn(10) * 0.1
        self.mask = xp.rint(self.rng.uniform(0.49, 1, size=10))

    def test_ista(self):
        it, x = lasso.solve(self.y, self.A, alpha=1.0, tol=1.0e-6,
                            method='ista', maxiter=1000)
        self.assertTrue(it < 1000 - 1)
        self.assert_minimum(x, alpha=1.0, tol=1.0e-5)

    def test_ista_mask(self):
        it, x = lasso.solve(self.y, self.A, alpha=1.0, tol=1.0e-6,
                            method='ista', maxiter=1000, mask=self.mask)
        self.assertTrue(it < 1000 - 1)
        self.assert_minimum(x, alpha=1.0, tol=1.0e-5, mask=self.mask)

    def test_fista(self):
        it, x = lasso.solve(self.y, self.A, alpha=1.0, tol=1.0e-6,
                            method='fista', maxiter=1000)
        self.assertTrue(it < 1000 - 1)
        self.assert_minimum(x, alpha=1.0, tol=1.0e-5)

    def test_fista_mask(self):
        it, x = lasso.solve(self.y, self.A, alpha=1.0, tol=1.0e-6,
                            method='fista', maxiter=1000, mask=self.mask)
        self.assertTrue(it < 1000 - 1)
        self.assert_minimum(x, alpha=1.0, tol=1.0e-5, mask=self.mask)

    def error(self, x, alpha, mask=None):
        mask = xp.ones_like(self.y) if mask is None else mask
        loss = xp.sum(xp.square(xp.abs(
                self.y - xp.tensordot(x, self.A, axes=1)) * mask))
        return 0.5 / alpha * loss + xp.sum(xp.abs(x))

    def assert_minimum(self, x, alpha, tol, n=3, mask=None):
        loss = self.error(x, alpha, mask)
        for _ in range(n):
            dx = self.randn(*x.shape) * tol
            self.assertTrue(loss < self.error(x + dx, alpha, mask))


class TestLassoMatrix(TestLasso):
    def setUp(self):
        self.rng = xp.random.RandomState(0)
        self.A = self.randn(5, 10)
        x_true = self.randn(55) * xp.rint(self.rng.uniform(size=55))
        self.x_true = x_true.reshape(11, 5)
        self.y = xp.dot(self.x_true,
                        self.A) + self.randn(11, 10) * 0.1
        v = self.rng.uniform(0.49, 1.0, size=110)
        v.resize(11, 10)
        self.mask = xp.rint(v)


class TestLassoTensor(TestLasso):
    def setUp(self):
        self.rng = xp.random.RandomState(0)
        self.A = self.randn(5, 10)
        x_true = self.randn(660) * xp.rint(self.rng.uniform(size=660))
        self.x_true = x_true.reshape(12, 11, 5)
        self.y = xp.dot(self.x_true,
                        self.A) + self.randn(12, 11, 10) * 0.1
        v = self.rng.uniform(0.49, 1, size=1320)
        v.resize(12, 11, 10)
        self.mask = xp.rint(v)


class TestLasso_complex(TestLasso):
    def randn(self, *shape):
        return self.rng.randn(*shape) + self.rng.randn(*shape) * 1.0j


class TestLasso_complexMatrix(TestLassoMatrix):
    def randn(self, *shape):
        return self.rng.randn(*shape) + self.rng.randn(*shape) * 1.0j


class TestLasso_complexTensor(TestLassoTensor):
    def randn(self, *shape):
        return self.rng.randn(*shape) + self.rng.randn(*shape) * 1.0j


if __name__ == '__main__':
    unittest.main()
