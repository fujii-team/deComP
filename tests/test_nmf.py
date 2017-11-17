import unittest
from decomp.utils.cp_compat import numpy_or_cupy as xp
from decomp.utils import normalize
import decomp.nmf as nmf
from .testings import allclose


class Test_NMF(unittest.TestCase):
    """ Test for Multiplicative Batch algorithm with L2 likilihood. """
    def randn(self, *shape):
        return self.rng.randn(*shape)

    def get_y(self, x, d):
        ytrue = xp.dot(x, d)
        if self.likelihood == 'l2':
            return ytrue + self.randn(*ytrue.shape) * 0.1
        elif self.likelihood == 'kl':  # KL
            return ytrue + xp.abs(self.randn(*ytrue.shape)) * 0.1
        raise NotImplementedError('Likelihood {} is not implemented'.format(
                                                            self.likelihood))

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
        if self.likelihood == 'l2':
            loss = xp.sum(xp.square(self.y - xp.dot(x, D)) * mask)
            return 0.5 * loss
        elif self.likelihood == 'kl':  # KL
            f = xp.maximum(xp.dot(x, D), 1.0e-15)
            return xp.sum((- self.y * xp.log(f) + f) * mask)
        raise NotImplementedError('Likelihood {} is not implemented'.format(
                                                            self.likelihood))

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


class FullbatchMixin(object):
    def run_fullbatch(self, mask):
        D = self.D.copy()
        it, D, x = nmf.solve(self.y, D, x=None, tol=1.0e-6,
                             minibatch=None, maxiter=self.maxiter,
                             method=self.method,
                             likelihood=self.likelihood, mask=mask,
                             random_seed=0)
        assert it < self.maxiter - 1
        self.assert_minimum(x, D, tol=1.0e-5, n=100, mask=mask)
        assert not allclose(D, xp.zeros_like(D), atol=1.0e-5)

    def test_run(self):
        self.run_fullbatch(mask=None)

    def test_run_mask(self):
        self.run_fullbatch(mask=self.mask)


class TestFullbatch_L2(Test_NMF, FullbatchMixin):
    @property
    def method(self):
        return 'mu'

    @property
    def likelihood(self):
        return 'l2'


class TestFullbatch_KL(TestFullbatch_L2):
    @property
    def likelihood(self):
        return 'kl'


class MinibatchMixin(object):
    def run_minibatch(self, mask):
        D = self.D.copy()
        start_iter = 100
        it, D, x = nmf.solve(
                     self.y, D, x=None, tol=1.0e-6,
                     minibatch=30, maxiter=start_iter,
                     method=self.method,
                     likelihood=self.likelihood, mask=mask,
                     random_seed=0)
        start_error = self.error(x, D, mask=mask)
        assert not allclose(D, xp.zeros_like(D), atol=1.0e-5)

        it, D, x = nmf.solve(
                     self.y, D, x=x, tol=1.0e-6,
                     minibatch=30, maxiter=self.maxiter,
                     method=self.method,
                     likelihood=self.likelihood, mask=mask,
                     random_seed=0)
        # just make sure the loss is decreasing
        assert self.error(x, D, mask=mask) < start_error
        assert not allclose(D, xp.zeros_like(D), atol=1.0e-5)

    def test_run_minibatch(self):
        self.run_minibatch(mask=None)

    def test_run_minibatch_mask(self):
        self.run_minibatch(mask=self.mask)

    @property
    def maxiter(self):
        return 300


class Test_ASG_MU_L2(Test_NMF, MinibatchMixin):
    @property
    def method(self):
        return 'asg-mu'

    @property
    def likelihood(self):
        return 'l2'


class Test_ASG_MU_KL(Test_ASG_MU_L2):
    @property
    def likelihood(self):
        return 'kl'


class Test_GSG_MU_L2(Test_NMF, MinibatchMixin):
    @property
    def method(self):
        return 'gsg-mu'

    @property
    def likelihood(self):
        return 'l2'


class Test_GSG_MU_KL(Test_GSG_MU_L2):
    @property
    def likelihood(self):
        return 'kl'


class Test_ASAG_MU_L2(Test_NMF, MinibatchMixin):
    @property
    def method(self):
        return 'asag-mu'

    @property
    def likelihood(self):
        return 'l2'


class Test_ASAG_MU_KL(Test_ASAG_MU_L2):
    @property
    def likelihood(self):
        return 'kl'


class Test_GSAG_MU_L2(Test_NMF, MinibatchMixin):
    @property
    def method(self):
        return 'gsag-mu'

    @property
    def likelihood(self):
        return 'l2'


class Test_GSAG_MU_KL(Test_GSAG_MU_L2):
    @property
    def likelihood(self):
        return 'kl'



if __name__ == '__main__':
    unittest.main()
