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


class PLSBase:
    def __init__(self, num: int = 1, defrate_y: bool = False, iter_num = 100):
        self.iter_num = 100
        self.num = num
        self.defrate_y = defrate_y
        self.x_weight = None
        self.y_weight = None
        self.x_score = None
        self.y_score = None

    @classmethod
    def _flip_wight(self, x, y, big=False):
        '''
        Set unique svd by flipping.
        '''
        if big:
            x *= np.sign(x[:, np.argmax(np.abs(x))])
            y *= np.sign(y[:, np.argmax(np.abs(y))])
        else:
            x *= np.sign(x[np.argmax(np.abs(x))])
            y *= np.sign(y[np.argmax(np.abs(y))])
        return x, y

    def _setup_array(self, x: int, y: int):
        self.x_weight = np.zeros((x.shape[-1], self.num))
        self.y_weight = np.zeros((y.shape[-1], self.num))
        self.x_score = np.zeros((x.shape[0], self.num))
        self.y_score = np.zeros((y.shape[0], self.num))

    @classmethod
    def center_scale(cls, x: np.ndarray, scale: bool = True):
        x = x - x.mean(axis=0)
        if scale:
            x = x / x.std(axis=0, ddof=1)
        return x

    @classmethod
    def _solve(self, x, y):
        return x.T @ y @ np.linalg.pinv(y.T @ y)

    def fit(self, x, y):
        self._setup_array(x, y)
        self.x, self.y = (self.center_scale(d) for d in (x, y))
        for c in range(self.num):
            self._current_num = c
            self._current_slice = slice(c, c+1)
            self._fit1(self.x, self.y)
        return self


class PLS(PLSBase):
    def _fit1(self, x, y):
        y_score = self.y[:, 0:1]
        for n in range(self.iter_num):
            x_weight = ((y_score.T @ self.x) / (y_score.T @ y_score)).T
            x_weight /= np.linalg.norm(x_weight)
            x_score = self.x @ x_weight
            y_weight = (self.y.T @ x_score) / (x_score.T @ x_score)
            y_weight /= np.linalg.norm(y_weight)
            y_score = self.y @ y_weight
        x_load = (x_score.T @ self.x) / (x_score.T @ x_score)
        y_load = (y_score.T @ self.y) / (y_score.T @ y_score)
        self.x -= np.dot(x_score , x_load)
        if self.defrate_y:
            self.y -= np.dot(y_score , y_load)
        x_weight, y_weight = self._flip_wight(x_weight, y_weight)
        self.x_weight[:, self._current_slice] = x_weight
        self.y_weight[:, self._current_slice] = y_weight
        self.x_score[:, self._current_slice] = x_score
        self.y_score[:, self._current_slice] = y_score
        return self

class NinPLSSVD(PLSBase):
    @classmethod
    def _get_weight_svd(self, x, y, flip=True):
        u, s, v = np.linalg.svd(x.T @ y, full_matrices=True)
        x_weight, y_weight = (n[:, 0:1] for n in (u, v.T))
        if flip:
            self._flip_wight(x_weight, y_weight)
        return x_weight, y_weight

    def fit(self, x, y, flip: bool = True):
        # self.x, self.y = (self.center_scale(d) for d in (x, y))
        # self.x_weight, s, self.y_weight = np.linalg.svd(self.x.T @ self.y)
        self.x, self.y = self._get_weight_svd(x, y, flip)
        # self._flip_wight(self.x_weight, self.y_weight, big=True)
        self.x_weight = self.x_weight[:, 0:1]
        self.y_weight = self.y_weight[:, 0:1]
        self.x_score = self.x @ self.x_weight
        self.y_score = self.y @ self.y_weight
        return self

ninpls = PLS(2)
# ninpls = PLS(2)
ninpls.fit(X, y)
print('xw', ninpls.x_weight)
print('yw', ninpls.y_weight)
print('xs', ninpls.x_score)
print('ys', ninpls.y_score)
# pls.fit(X, y, svd=False)
pls = PLSSVD(2, scale=True)
pls = PLSRegression(2, scale=True)
pls.fit(X, y)
np.testing.assert_almost_equal(ninpls.x_weight, pls.x_weights_, decimal = 5)
np.testing.assert_almost_equal(ninpls.x_score, pls.x_scores_, decimal = 5)
np.testing.assert_almost_equal(ninpls.x_load, pls.x_loadings_, decimal = 5)
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

