"""Thanks to Tim Roith and Samira Kabri at DESY who helped creating an earlier version of this notebook."""

import numpy as np
import skimage as ski


def soft_shrinkage(x, lamda):
    return np.maximum(np.abs(x) - lamda, 0.0) * np.sign(x)


class Identity:
    def __call__(self, u):
        return u

    def adjoint(self, u):
        return u

    def inv(self, u):
        return u


class Radon:
    def __init__(self, theta=None):
        self.theta = theta if theta is not None else np.linspace(0, 180, 50)
        self.num_theta = len(self.theta)

    def __call__(self, u):
        return ski.transform.radon(u, self.theta) / u.shape[-1]

    def adjoint(self, k):
        return ski.transform.iradon(k, self.theta, filter_name=None) / (
            k.shape[0] * np.pi / (2 * self.num_theta)
        )

    def inv(self, k):
        return ski.transform.iradon(k * k.shape[0], self.theta) # , filter_name=None for no filter in FBP

    inverse = inv
    T = adjoint


def test_adjoint(A, x, y=None):
    Ax = A(x)
    if y is None:
        y = np.random.uniform(size=Ax.shape)
    res_1 = np.sum(Ax * y)
    res_2 = np.sum(x * A.adjoint(y))
    return res_1, res_2
