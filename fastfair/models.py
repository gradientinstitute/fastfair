# Copyright 2023 Gradient Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fair regression models using MI lower bound approximation."""

import jax.numpy as np
from jax import grad, jit, random
from scipy.optimize import minimize
from numpy.random import RandomState
from sklearn.base import BaseEstimator
import time


SPSCALER = 30  # Make the softplus function closer to max(x, 0)
MINF = 1e-10


#
# Private module functions
#

def _predict(X, w):
    s = np.dot(X, w[1:]) + w[0]
    return s


def _softplus(x):
    sp = np.logaddexp(0., SPSCALER * x) / SPSCALER
    return sp


def _quadbasis(X):
    N, D = X.shape
    B = (X[:, np.newaxis, :] * X[:, :, np.newaxis]).reshape((N, D**2))
    BX = np.hstack((B, X))
    return BX


def _lspc(X, a, lambda_v, quadbasis, V=None):
    """Least squares probabilistic classifier implementation."""

    # Quadratic basis for non-linearity
    if quadbasis:
        X = _quadbasis(X)

    # add a bias term to X
    N, D = X.shape[0], X.shape[1] + 1
    Xb = np.hstack((np.ones((N, 1)), X))

    # Learn classifier weights
    if V is None:
        assert a is not None
        A = np.dot(Xb.T, Xb) + np.eye(D) * lambda_v
        b = np.dot(Xb.T, a)
        V = np.linalg.solve(A, b)

    # log probabilities, finite samples size correction
    f = np.dot(Xb, V)
    # qa = np.maximum(MINF, f)
    qa = _softplus(f)  # this is numerically better than maximum(f, 0)
    log_pa = np.log(qa) - np.log(np.sum(qa, axis=1))[:, np.newaxis]

    return log_pa, V


#
# Public module functions/classes
#

class Linear(BaseEstimator):
    def __init__(self, lambda_w=.1, opt_tol=1e-4, random_state=None):
        self.lambda_w = lambda_w
        self.w = None
        self.opt_tol = opt_tol
        self.random_state = random_state
        if random_state is None:
            self.random_state = RandomState().randint(-100000, 100000)
        random.PRNGKey(self.random_state)

    def predict(self, X):
        s = _predict(X, self.w)
        return s

    def fit(self, X, y, a=None):
        # Make sure a is categorical
        if np.ndim(a) == 1:
            a = np.vstack((1 - a, a)).T

        # loss and gradient
        loss = jit(self._loss(X, y, a))
        dloss = grad(loss)

        # Estimate W0 (ols)
        X1 = np.hstack((np.ones((X.shape[0], 1)), X))
        eps = 1e-9
        w_0 = np.linalg.solve(np.dot(X1.T, X1) + eps * np.eye(X1.shape[1]),
                              np.dot(X1.T, y))

        t0 = time.time()
        # Optimise
        res = minimize(loss, w_0, jac=dloss, tol=self.opt_tol)

        self.opt_time = time.time() - t0
        self.w = res.x
        self.calls = res.nfev
        return self

    def _loss(self, X, y, a):
        def loss(w):
            l = self._loss_linear(X, y, w)
            return l
        return loss

    def _loss_linear(self, X, y, w):
        s = _predict(X, w)
        sqerror = (y - s)**2
        reg = np.dot(w, w)
        loss = np.mean(sqerror) + self.lambda_w * reg
        return loss

    def get_params(self, deep=True):
        return {"lambda_w": self.lambda_w, "random_state": self.random_state}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class LSPCIndependence(Linear):

    def __init__(self, lambda_w=.1, lambda_f=.1, lspc_lambda=1.,
                 quadbasis=False, opt_tol=1e-4, random_state=None):
        super().__init__(lambda_w, opt_tol=opt_tol, random_state=random_state)
        self.lambda_f = lambda_f
        self.lspc_lambda = lspc_lambda
        self.quadbasis = quadbasis

    def _loss(self, X, y, a):
        def lossfn(w):
            loss = self._loss_linear(X, y, w) + self._loss_indep(X, a, w)
            return loss
        return lossfn

    def _loss_indep(self, X, a, w):
        N = len(X)
        s = _predict(X, w)
        qa, _ = _lspc(s[:, np.newaxis], a, self.lspc_lambda, self.quadbasis)
        logp = a * qa
        loss = self.lambda_f * np.sum(logp) / N
        return loss

    def get_params(self, deep=True):
        return {
            "lambda_w": self.lambda_w,
            "lambda_f": self.lambda_f,
            "lspc_lambda": self.lspc_lambda,
            "quadbasis": self.quadbasis,
            "random_state": self.random_state
        }


class LSPCSeparation(LSPCIndependence):

    def _loss(self, X, y, a):
        def lossfn(w):
            loss = self._loss_linear(X, y, w) + self._loss_separ(X, y, a, w)
            return loss
        return lossfn

    def _loss_separ(self, X, y, a, w):
        N = len(X)
        s = _predict(X, w)
        sy = np.vstack((s, y)).T
        qa, _ = _lspc(sy, a, self.lspc_lambda, self.quadbasis)
        logp = a * qa
        loss = self.lambda_f * np.sum(logp) / N
        return loss


class LSPCSufficiency(LSPCSeparation):

    def _loss(self, X, y, a):
        def lossfn(w):
            loss_suf = self._loss_separ(X, y, a, w) \
                - self._loss_indep(X, a, w)
            loss = self._loss_linear(X, y, w) + loss_suf
            return loss
        return lossfn


class LSPClassifier(BaseEstimator):

    def __init__(self, lambda_v=1., quadbasis=False):
        self.lambda_v = lambda_v
        self.quadbasis = quadbasis

    def fit(self, X, y):
        # Make sure y is categorical
        if np.ndim(y) == 1:
            y = np.vstack((1 - y, y)).T

        self.classes_ = np.arange(np.ndim(y))

        _, self.V = _lspc(X, y, lambda_v=self.lambda_v,
                          quadbasis=self.quadbasis)
        return self

    def predict(self, X):
        log_p, _ = _lspc(X, None, lambda_v=self.lambda_v,
                         quadbasis=self.quadbasis, V=self.V)
        y_hat = np.argmax(log_p, axis=1)
        return y_hat

    def predict_proba(self, X):
        log_p, _ = _lspc(X, None, lambda_v=self.lambda_v,
                         quadbasis=self.quadbasis, V=self.V)
        p = np.exp(log_p)
        return p

    def get_params(self, deep=True):
        return {"lambda_v": self.lambda_v, "quadbasis": self.quadbasis}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
