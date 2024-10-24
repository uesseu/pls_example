from logging import basicConfig, ERROR
from sklearn.cross_decomposition import PLSRegression, PLSSVD
import numpy as np
basicConfig(level=ERROR)


def is_smallerthan_atom(x) -> bool:
    return x < np.finfo(x.dtype).eps


class PLSBase:
    def __init__(self, num: int = 1, defrate_y: bool = False,
                 iter_num: int = 100):
        self.iter_num = 500
        self.num = num
        self.defrate_y = defrate_y
        self.x_weight = None
        self.y_weight = None
        self.x_score = None
        self.y_score = None

    @classmethod
    def _flip_weight(self, params, big=False) -> None:
        '''
        Set unique svd by flipping.
        '''
        if big:
            sign = np.sign(params[0][np.argmax(np.abs(params[0].mean(1)))])
        else:
            sign = np.sign(params[0][np.argmax(np.abs(params[0]))])
        for p in params:
            p *= sign

    def _setup_array(self, x: int, y: int) -> None:
        self.x_weight = np.zeros((x.shape[-1], self.num))
        self.y_weight = np.zeros((y.shape[-1], self.num))
        self.x_score = np.zeros((x.shape[0], self.num))
        self.y_score = np.zeros((y.shape[0], self.num))

    @classmethod
    def center_scale(cls, x: np.ndarray, scale: bool = True) -> np.ndarray:
        x = x - x.mean(axis=0)
        if scale:
            x = x / x.std(axis=0, ddof=1)
        return x

    def fit(self, x, y) -> 'PLSBase':
        self._setup_array(x, y)
        self.x, self.y = (self.center_scale(d) for d in (x, y))
        for c in range(self.num):
            self._current_num = c
            self._current_slice = slice(c, c+1)
            self._fit1(self.x, self.y)
        return self

    def _put_params(self, x_weight, y_weight, x_score, y_score) -> None:
        self.x_weight[:, self._current_slice] = x_weight
        self.y_weight[:, self._current_slice] = y_weight
        self.x_score[:, self._current_slice] = x_score
        self.y_score[:, self._current_slice] = y_score


class NinPLS(PLSBase):
    def _fit1(self, x, y) -> 'NinPLS':
        y_score = self.y[:, 0:1]
        for n in range(self.iter_num):
            # x_weight = ((y_score.T @ self.x) / (y_score.T @ y_score)).T
            x_weight = (y_score.T @ self.x).T
            x_weight /= np.linalg.norm(x_weight)
            x_score = x @ x_weight
            # y_weight = (self.y.T @ x_score) / (x_score.T @ x_score)
            y_weight = self.y.T @ x_score
            y_weight /= np.linalg.norm(y_weight)
            y_score = y @ y_weight
        self._flip_weight([x_weight, y_weight, x_score, y_score])
        x_load = (x_score.T @ self.x) / (x_score.T @ x_score)
        y_load = (y_score.T @ self.y) / (y_score.T @ y_score)
        self.x -= np.dot(x_score, x_load)
        if self.defrate_y:
            self.y -= np.dot(y_score, y_load)
        self._put_params(x_weight, y_weight, x_score, y_score)
        return self


class NinPLSSVD1(PLSBase):
    def _fit1(self, x, y) -> 'NinPLSSVD1':
        u, s, v = np.linalg.svd(x.T @ y, full_matrices=True)
        x_weight, y_weight = (n[:, 0:1] for n in (u, v.T))
        x_score = x @ x_weight
        y_score = y @ y_weight
        self._flip_weight([x_weight, y_weight, x_score, y_score])
        x_load = (x_score.T @ self.x) / (x_score.T @ x_score)
        y_load = (y_score.T @ self.y) / (y_score.T @ y_score)
        self.x -= x_score @ x_load
        if self.defrate_y:
            self.y -= np.dot(y_score, y_load)
        self._put_params(x_weight, y_weight, x_score, y_score)
        return self


class NinPLSSVD(PLSBase):
    def fit(self, x, y, flip: bool = True) -> 'NinPLSSVD':
        x, y = (self.center_scale(d) for d in (x, y))
        u, s, v = np.linalg.svd(x.T @ y, full_matrices=True)
        self.x_weight, self.y_weight = (x[:, 0: self.num] for x in (u, v.T))
        if flip:
            self._flip_weight([self.x_weight, self.y_weight], True)
        self.x_score = x @ self.x_weight
        self.y_score = y @ self.y_weight
        return self


X = np.array(
    [[3., 0., 1.],
     [1., 2., 0.],
     [2., 8., 2.],
     [3., 5., 4.]]
)
y = np.array(
    [[0.1, -0.2],
     [0.9, 1.1],
     [3.2, 5.9],
     [6.9, 12.3]]
)


def is_good(ninpls_cls, pls_cls, score: bool = True):
    ninpls = ninpls_cls(2).fit(X, y)
    pls = pls_cls(2).fit(X, y)
    np.testing.assert_almost_equal(ninpls.x_weight, pls.x_weights_, decimal=5)
    if score:
        np.testing.assert_almost_equal(ninpls.x_score,
                                       pls.x_scores_, decimal=5)

ninpls = NinPLS(2)
ninpls = NinPLSSVD1(2)
ninpls.fit(X, y)
pls = PLSSVD(2, scale=True)
pls = PLSRegression(2, scale=True)
pls.fit(X, y)

is_good(NinPLS, PLSRegression)
is_good(NinPLSSVD1, PLSRegression)
is_good(NinPLSSVD, PLSSVD, False)
