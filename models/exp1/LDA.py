import numpy as np


class LinearDiscriminantAnalysis:
    """线性判别分析。

    参数
    ----
    n_components : int or None
        保留的判别方向数；None 时取 min(K-1, n_features)。

    属性
    ----
    classes_ : ndarray, shape (K,)
    means_ : ndarray, shape (K, n_features)
    priors_ : ndarray, shape (K,)
    covariance_ : ndarray, shape (n_features, n_features)
        各类共享的池化协方差矩阵。
    scalings_ : ndarray, shape (n_features, n_components)
        投影矩阵（各列为 Fisher 判别方向，按特征值从大到小排列）。
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        K, (n, d) = self.classes_.size, X.shape

        self.means_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.priors_ = np.array([np.sum(y == c) / n for c in self.classes_])

        mu = X.mean(axis=0)
        Sw = np.zeros((d, d))
        Sb = np.zeros((d, d))
        for c, m in zip(self.classes_, self.means_):
            Xc = X[y == c] - m
            Sw += Xc.T @ Xc                                   # 类内散度
            diff = (m - mu).reshape(-1, 1)
            Sb += np.sum(y == c) * (diff @ diff.T)             # 类间散度
        self.covariance_ = Sw / (n - K)                        # 池化协方差

        # 广义特征值问题 Sb v = λ Sw v：先对 Sw 做 Cholesky 白化，转为对称问题
        L = np.linalg.cholesky(Sw)
        L_inv = np.linalg.inv(L)
        M = L_inv @ Sb @ L_inv.T
        evals, U = np.linalg.eigh(M)                           # 对称矩阵，升序
        order = np.argsort(evals)[::-1]
        evals, U = evals[order], U[:, order]
        n_comp = min(K - 1, d) if self.n_components is None else self.n_components
        V = L_inv.T @ U[:, :n_comp]                           # 还原到原空间
        # 统一符号：让每个方向中绝对值最大的分量为正，便于与调包实现对照
        for j in range(V.shape[1]):
            if V[np.argmax(np.abs(V[:, j])), j] < 0:
                V[:, j] *= -1
        self.scalings_ = V
        self.explained_variance_ratio_ = evals[:n_comp] / evals[:n_comp].sum()
        return self

    # ------------------------------------------------------------------
    # 投影与判别
    # ------------------------------------------------------------------
    def transform(self, X):
        return np.asarray(X, dtype=float) @ self.scalings_

    def _discriminant(self, X):
        """各类的高斯判别函数值 δ_k(x)。"""
        X = np.asarray(X, dtype=float)
        Sinv = np.linalg.pinv(self.covariance_)
        M = Sinv @ self.means_.T                               # (d, K)
        return X @ M - 0.5 * np.sum(self.means_.T * M, axis=0) + np.log(self.priors_)

    def decision_function(self, X):
        return self._discriminant(X)

    def predict(self, X):
        return self.classes_[np.argmax(self._discriminant(X), axis=1)]

    def predict_proba(self, X):
        Z = self._discriminant(X)
        Z = Z - Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        return P / P.sum(axis=1, keepdims=True)

    def get_params(self, deep=True):
        return {"n_components": self.n_components}
