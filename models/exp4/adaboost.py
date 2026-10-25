import copy

import numpy as np
from scipy import sparse


class DecisionStump:
    """决策树桩：在某一特征上取一个阈值，把样本分成左右两部分并分别给一个类别。"""

    def __init__(self, random_state=None):
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X = X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int).ravel()
        n, n_features = X.shape
        w = (np.ones(n) if sample_weight is None
             else np.asarray(sample_weight, dtype=float).ravel())
        total_w = w.sum()

        best = (np.inf, None, None, 0, 1)
        for f in range(n_features):
            order = np.argsort(X[:, f], kind="mergesort")
            xs, ys, ws = X[order, f], y[order], w[order]
            cum_pos = np.concatenate([[0.0], np.cumsum(np.where(ys == 1, ws, 0.0))])
            cum_w = np.concatenate([[0.0], np.cumsum(ws)])
            valid = np.where(xs[:-1] < xs[1:])[0]
            if valid.size == 0:
                continue
            left_pos = cum_pos[valid + 1]
            left_w = cum_w[valid + 1]
            right_pos = cum_pos[-1] - left_pos
            right_w = cum_w[-1] - left_w
            # 极性 A：左判 0、右判 1；极性 B：左判 1、右判 0
            err_a = left_pos + (right_w - right_pos)
            err_b = (left_w - left_pos) + right_pos
            for errs, left_label, right_label in ((err_a, 0, 1), (err_b, 1, 0)):
                k = int(np.argmin(errs))
                e = float(errs[k])
                if e < best[0]:
                    thr = 0.5 * (xs[valid[k]] + xs[valid[k] + 1])
                    best = (e, f, thr, left_label, right_label)

        self._err, self.feature_, self.threshold_ = best[0], best[1], best[2]
        self.left_label_, self.right_label_ = best[3], best[4]
        self.weighted_error_ = self._err / total_w
        return self

    def predict(self, X):
        X = X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=float)
        left = X[:, self.feature_] <= self.threshold_
        return np.where(left, self.left_label_, self.right_label_)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))


class AdaBoostClassifier:
    """AdaBoost（二分类）。

    参数
    ----
    estimator : object
        弱学习器，需支持 fit(X, y, sample_weight) 与 predict(X)。
    n_estimators : int
        弱学习器个数。
    learning_rate : float
        学习率，用于缩放每个弱学习器的权重 α。
    random_state : int or None
        预留参数，保证接口与 sklearn 一致。
    """

    def __init__(self, estimator=None, n_estimators=50, learning_rate=1.0,
                 random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X = X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("本实现仅支持二分类")
        yy = np.where(y == self.classes_[0], -1.0, 1.0)

        n = X.shape[0]
        w = np.full(n, 1.0 / n) if sample_weight is None else np.asarray(sample_weight, float).copy()
        w /= w.sum()

        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []

        for _ in range(self.n_estimators):
            h = copy.deepcopy(self.estimator)
            h.fit(X, y, sample_weight=w)
            pred = h.predict(X)
            err = float(np.sum(w * (pred != y)) / w.sum())
            # sklearn 的处理：错误率过高（≥ 1 - 1/K）时停止；过小时裁剪以防数值问题
            if err >= 1.0 - 1.0 / self.classes_.size:
                break
            err = max(err, 1e-10)
            alpha = self.learning_rate * 0.5 * np.log((1.0 - err) / err)
            w = w * np.exp(-alpha * yy * np.where(pred == self.classes_[0], -1.0, 1.0))
            w /= w.sum()
            self.estimators_.append(h)
            self.estimator_weights_.append(alpha)
            self.estimator_errors_.append(err)

        self.n_estimators_ = len(self.estimators_)
        return self

    def decision_function(self, X):
        X = X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=float)
        s = np.zeros(X.shape[0])
        for h, a in zip(self.estimators_, self.estimator_weights_):
            pred = h.predict(X)
            s += a * np.where(pred == self.classes_[0], -1.0, 1.0)
        return s

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    def get_params(self, deep=True):
        return {"estimator": self.estimator, "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate, "random_state": self.random_state}
