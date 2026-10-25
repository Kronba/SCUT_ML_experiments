import numpy as np


def sigmoid(z):
    """Sigmoid 函数 σ(z) = 1 / (1 + exp(-z))，对 exp 溢出做了分段处理。"""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos, neg = z >= 0, z < 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


class LogisticRegression:
    """二分类逻辑斯蒂回归。

    参数
    ----
    solver : {"newton", "gd"}
        newton 为牛顿法（二阶收敛，迭代少），gd 为梯度下降。
    learning_rate : float
        梯度下降的学习率。
    n_iters : int
        最大迭代次数。
    tol : float
        收敛阈值：梯度的无穷范数小于该值即停止。
    C : float
        正则化系数的倒数，与 scikit-learn 的含义一致（C 越大正则越弱，C 越小正则越强）。

    属性
    ----
    coef_ : ndarray, shape (1, n_features)
    intercept_ : ndarray, shape (1,)
    loss_history_ : list[float]
    n_iter_ : int
    """

    def __init__(self, solver="newton", learning_rate=0.1, n_iters=200,
                 tol=1e-8, C=1.0):
        self.solver = solver
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.tol = tol
        self.C = C

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _add_bias(X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _loss(self, theta, Xb, y):
        """目标函数：J(θ) = 0.5‖w‖² + C·Σ[y·(-ln p) + (1-y)·(-ln(1-p))]。"""
        p = sigmoid(Xb @ theta)
        eps = 1e-12
        data_loss = -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        return 0.5 * np.sum(theta[1:] ** 2) + self.C * data_loss

    def _grad_hess(self, theta, Xb, y):
        """返回目标函数在 θ 处的梯度与海森矩阵。"""
        p = sigmoid(Xb @ theta)
        r = p - y                                   # 残差 p - y
        grad = np.empty_like(theta)
        grad[0] = self.C * np.sum(r)                # 截距项
        grad[1:] = theta[1:] + self.C * (Xb[:, 1:].T @ r)
        s = p * (1 - p)                             # σ'(z)
        Xs = Xb * s[:, None]
        hess = self.C * (Xb.T @ Xs)
        hess[1:, 1:] += np.eye(Xb.shape[1] - 1)     # L2 正则的二阶项
        return grad, hess

    # ------------------------------------------------------------------
    # 训练与预测
    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("本实现仅支持二分类")
        Xb = self._add_bias(X)
        theta = np.zeros(Xb.shape[1])
        self.loss_history_ = [self._loss(theta, Xb, y)]

        for it in range(1, self.n_iters + 1):
            grad, hess = self._grad_hess(theta, Xb, y)
            if np.linalg.norm(grad, ord=np.inf) < self.tol:
                self.n_iter_ = it - 1
                break
            if self.solver == "newton":
                step = np.linalg.solve(hess, grad)
            elif self.solver == "gd":
                step = self.learning_rate * grad
            else:
                raise ValueError("solver 只能是 'newton' 或 'gd'")
            theta = theta - step
            self.loss_history_.append(self._loss(theta, Xb, y))
            if np.linalg.norm(step, ord=np.inf) < self.tol:
                self.n_iter_ = it
                break
        else:
            self.n_iter_ = self.n_iters

        self.intercept_ = np.array([theta[0]])
        self.coef_ = theta[1:].reshape(1, -1)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_.ravel() + self.intercept_[0]

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {"solver": self.solver, "learning_rate": self.learning_rate,
                "n_iters": self.n_iters, "tol": self.tol, "C": self.C}
