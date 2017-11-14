import unittest
from decomp.utils.cp_compat import numpy_or_cupy as xp
import decomp.nmf as nmf
from .testings import allclose


class TestFloat(unittest.TestCase):
    def randn(self, *shape):
        return self.rng.randn(*shape)

    def setUp(self):
        self.rng = xp.random.RandomState(0)
        self.Dtrue = xp.maximum(self.randn(3, 5), 0.0)
        self.xtrue = xp.maximum(self.randn(101, 3), 0.0)

        self.y = xp.dot(self.xtrue, self.Dtrue) + self.randn(101, 5) * 0.1
        self.D = xp.maximum(self.Dtrue + self.randn(*self.Dtrue.shape) * 0.3,
                            0.0)
        self.mask = xp.rint(
                self.rng.uniform(0.3, 1, size=505)).reshape(101, 5)

    def error(self, x, D, mask=None):
        mask = xp.ones(self.y.shape, dtype=float) if mask is None else mask
        D = D / xp.maximum(xp.sum(D * D, axis=0), 1.0)
        loss = xp.sum(xp.square(xp.abs(
            self.y - xp.tensordot(x, D, axes=1))))
        return 0.5 * loss

    def assert_minimum(self, x, D, tol, n=100, mask=None):
        loss = self.error(x, D, mask)
        for _ in range(n):
            dx = self.randn(*x.shape) * tol
            dD = self.randn(*D.shape) * tol
            assert loss < self.error(xp.maximum(x + dx, 0.0),
                                     xp.maximum(D + dD, 0.0), mask) + tol

    @property
    def likelihood(self):
        return 'square'

    def test_run(self):
        D = self.D.copy()
        it, D, x = nmf.solve(self.y, D, minibatch=None, maxiter=1000,
                             tol=1.0e-6, random_seed=0)
        assert it < 1000 - 1
        print(self.D)
        self.assert_minimum(x, D, tol=1.0e-5, n=100)
        assert not allclose(D, xp.zeros_like(D), atol=1.0e-5)

        D_minibatch = self.D.copy()
        it, D_minibatch, x_minibatch = nmf.solve(
            self.y, D, minibatch=10, maxiter=1000, tol=1.0e-6, random_seed=0)

        assert it < 1000 - 1
        assert allclose(D, D_minibatch, atol=1.0e-5)

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


if __name__ == '__main__':
    unittest.main()
