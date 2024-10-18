from logging import basicConfig, INFO, ERROR
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression, PLSSVD, PLSCanonical
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import numpy as np
basicConfig(level=ERROR)

X = np.array(
    [[3., 0., 1.],
     [1.,2.,0.],
     [2.,8.,2.],
     [3.,5.,4.]]
)
y = np.array(
    [[0.1, -0.2],
     [0.9, 1.1],
     [3.2, 5.9],
     [6.9, 12.3]]
)
# y = y[:, 0:1]
# y = y[0:1, :]

class PLSBase:
    def __init__(self, num: int = 1):
        self.num = num
        self.x_weight = []
        self.y_weight = []
        self.x_score = []
        self.y_score = []

    def _setup_array(self, x, y):
        self.x_weight = np.zeros((self.num, x.shape[-1]))
        self.y_weight = np.zeros((self.num, x.shape[-1]))
        self.x_weight = np.zeros((self.num, x.shape[-1]))
        self.y_weight = np.zeros((self.num, x.shape[-1]))

    def defrate(self):
        self.x -= np.outer(self.x_score, self.x_load)
        self.y -= np.outer(self.y_score, self.y_load)

    @classmethod
    def center_scale(cls, x: np.ndarray, scale: bool = True):
        x = x - x.mean(axis=0)
        if scale:
            x = x / x.std(axis=0, ddof=1)
        return x

    @classmethod
    def _solve(self, x, y):
        return x.T @ y / (y.T @ y)

    def fit(self, x, y):
        self.x, self.y = (self.center_scale(d) for d in (x, y))
        for c in range(self.num):
            self._fit1(self.x, self.y)
        return self


class PLS(PLSBase):
    @classmethod
    def _get_weight_pls1(self, n):
        return n / np.linalg.norm(n, axis=0)

    def _fit1(self, x, y, svd=False):
        x_weight = self._get_weight_pls1(self.x.T @ self.y)
        x_score = self.x @ self.x_weight
        y_weight = self._get_weight_pls1(self.y.T @ x_score)
        y_weight /= np.sqrt(y_weight @ y_weight)
        y_score = self.y @ self.y_weight
        x_load = self._solve(self.x, x_score)
        y_load = self._solve(self.y, y_score)
        x_coef = (y_score.T @ x_score) / (x_score.T @ x_score)
        self.x -= np.outer(x_score, x_load)
        self.y -= np.outer(y_score, y_load)
        self.x_weight.append(x_weight)
        self.y_weight.append(y_weight)
        self.x_score.append(x_score)
        self.y_score.append(y_score)
        return self

class PLSSVD(PLSBase):
    @classmethod
    def _get_weight_svd(self, x, y, flip=True):
        u, s, v = np.linalg.svd(x.T @ y, full_matrices=True)
        x_weight, y_weight = (n[:, 0:1] for n in (u, v.T))
        if flip:
            self._flip_wight(x_weight, y_weight)
        return x_weight, y_weight

    @classmethod
    def _flip_wight(self, x, y):
        '''
        Set unique svd by flipping.
        '''
        sign = np.sign(x[np.argmax(np.abs(x))])
        x *= sign
        y *= sign

    def _fit1(self, x, y, svd=False):
        x_weight, y_weight = self._get_weight_svd(self.x, self.y)
        x_score = self.x @ x_weight
        y_score = self.y @ y_weight
        x_load = self._solve(self.x, x_score)
        y_load = self._solve(self.y, y_score)
        x_coef = (y_score.T @ x_score) / (x_score.T @ x_score)
        self.x -= np.outer(x_score, x_load)
        self.y -= np.outer(y_score, y_load)
        self.x_weight.append(x_weight)
        self.y_weight.append(y_weight)
        self.x_score.append(x_score)
        self.y_score.append(y_score)
        return self

ninpls = PLSSVD(2)
ninpls.fit(X, y)
print('xw', ninpls.x_weight)
print('yw', ninpls.y_weight)
print('xs', ninpls.x_score)
print('ys', ninpls.y_score)
# pls.fit(X, y, svd=False)
pls = PLSRegression(1, scale=True)
pls.fit(X, y)

np.testing.assert_almost_equal(ninpls.x_weight, pls.x_weights_)
np.testing.assert_almost_equal(ninpls.x_score, pls.x_scores_)
np.testing.assert_almost_equal(ninpls.x_load, pls.x_loadings_)
print(pls.y_scores_)
print(ninpls.y_score)
print('x', ninpls.x)
# np.testing.assert_almost_equal(ninpls.y_score, pls.y_scores_)
print('yw', ninpls.y_weight, pls.y_weights_)
# np.testing.assert_almost_equal(ninpls.y_weight, pls.y_weights_)
# print('ys', ninpls.y_score, pls.y_scores_)
# print('y', ninpls.y, pls._y_mean)
# np.testing.assert_almost_equal(ninpls.y_weight, pls.y_weights_)
exit()
print(ninpls.y_load, pls.y_loadings_)

