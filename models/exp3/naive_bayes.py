import numpy as np
from scipy import sparse


class MultinomialNB:
    """多项式朴素贝叶斯。

    参数
    ----
    alpha : float
        拉普拉斯平滑系数，避免未出现的词把条件概率估成 0。
    fit_prior : bool
        是否从数据中估计先验概率；为 False 时使用均匀先验。
    class_prior : array-like or None
        手动指定先验概率。
    """

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def fit(self, X, y):
        X = sparse.csr_matrix(X) if sparse.issparse(X) else np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        n_classes, n_features = self.classes_.size, X.shape[1]

        feature_count = np.zeros((n_classes, n_features))
        class_count = np.zeros(n_classes)
        for k, c in enumerate(self.classes_):
            Xc = X[y == c]
            feature_count[k] = np.asarray(Xc.sum(axis=0)).ravel()
            class_count[k] = Xc.shape[0]

        self.feature_count_ = feature_count
        self.class_count_ = class_count

        # 拉普拉斯平滑后的类条件概率（取对数）
        smoothed = feature_count + self.alpha
        self.feature_log_prob_ = np.log(smoothed) - np.log(
            smoothed.sum(axis=1, keepdims=True))

        # 先验概率
        if self.class_prior is not None:
            prior = np.asarray(self.class_prior, dtype=float)
        elif self.fit_prior:
            prior = class_count / class_count.sum()
        else:
            prior = np.full(n_classes, 1.0 / n_classes)
        self.class_log_prior_ = np.log(prior)
        return self

    def _joint_log_likelihood(self, X):
        X = sparse.csr_matrix(X) if sparse.issparse(X) else np.asarray(X, dtype=float)
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]

    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(X)
        # 对数归一化：log p_k = jll_k - log sum exp(jll)
        m = jll.max(axis=1, keepdims=True)
        log_norm = m + np.log(np.exp(jll - m).sum(axis=1, keepdims=True))
        return jll - log_norm

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    def get_params(self, deep=True):
        return {"alpha": self.alpha, "fit_prior": self.fit_prior,
                "class_prior": self.class_prior}
