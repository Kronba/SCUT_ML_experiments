import numpy as np


class Perceptron:
    """线性感知机。

    参数
    ----
    learning_rate : float
        学习率 η；发现误分类样本时按 w ← w + η·y·x 更新。
    max_iter : int
        最大迭代轮数。
    tol : float
        提前停止的阈值：若连续若干轮的误分类数下降幅度小于该值则停止。
    shuffle : bool
        每轮是否打乱样本顺序。
    random_state : int or None
        随机种子，保证结果可复现。
    """

    def __init__(self, learning_rate=1.0, max_iter=1000, tol=1e-3,
                 shuffle=True, random_state=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("本实现仅支持二分类")
        yy = np.where(y == self.classes_[0], -1.0, 1.0)
        m, n = X.shape
        rng = np.random.default_rng(self.random_state)

        w = np.zeros(n)
        b = 0.0
        best_err = np.inf
        no_improve = 0
        self.errors_ = []

        for epoch in range(self.max_iter):
            order = rng.permutation(m) if self.shuffle else np.arange(m)
            for idx in order:
                # 误分类判定：y_i (w·x_i + b) ≤ 0
                if yy[idx] * (X[idx] @ w + b) <= 0:
                    w += self.learning_rate * yy[idx] * X[idx]
                    b += self.learning_rate * yy[idx]
            err = np.mean((X @ w + b >= 0) != (yy > 0))
            self.errors_.append(err)
            self.n_iter_ = epoch + 1
            # 提前停止：连续 5 轮没有明显改进
            if best_err - err < self.tol:
                no_improve += 1
                if no_improve >= 5:
                    break
            else:
                no_improve = 0
            best_err = min(best_err, err)

        self.coef_ = w.reshape(1, -1)
        self.intercept_ = np.array([b])
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_.ravel() + self.intercept_[0]

    def predict(self, X):
        z = self.decision_function(X)
        return np.where(z >= 0, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    def get_params(self, deep=True):
        return {"learning_rate": self.learning_rate, "max_iter": self.max_iter,
                "tol": self.tol, "shuffle": self.shuffle,
                "random_state": self.random_state}
