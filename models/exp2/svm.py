import numpy as np


class SVC:
    """支持向量分类器（SMO 求解）。

    参数
    ----
    C : float
        惩罚系数，越大越不容忍违反间隔的样本。
    kernel : {"rbf", "linear", "poly"}
    gamma : {"scale"} or float
        高斯核与多项式核的参数；"scale" 取 1/(n_features * X.var())，与 sklearn 一致。
    degree : int
        多项式核的次数。
    coef0 : float
        多项式核的常数项。
    tol : float
        KKT 条件的容差，用于判断是否停止迭代。
    max_iter : int
        最大迭代次数。
    """

    def __init__(self, C=1.0, kernel="rbf", gamma="scale", degree=3, coef0=0.0,
                 tol=1e-3, max_iter=2000):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter

    # ------------------------------------------------------------------
    def _resolve_gamma(self, X):
        if isinstance(self.gamma, str):
            if self.gamma == "scale":
                return 1.0 / (X.shape[1] * X.var())
            if self.gamma == "auto":
                return 1.0 / X.shape[1]
            raise ValueError("gamma 只能是 'scale'、'auto' 或数值")
        return float(self.gamma)

    def _kernel_matrix(self, A, B):
        """计算核矩阵 K(A, B)。"""
        if self.kernel == "linear":
            return A @ B.T
        if self.kernel == "poly":
            return (self._gamma_ * (A @ B.T) + self.coef0) ** self.degree
        if self.kernel == "rbf":
            sq = (np.sum(A ** 2, axis=1)[:, None]
                  + np.sum(B ** 2, axis=1)[None, :]
                  - 2.0 * (A @ B.T))
            return np.exp(-self._gamma_ * np.maximum(sq, 0.0))
        raise ValueError("kernel 只能是 'rbf'、'linear' 或 'poly'")

    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("本实现仅支持二分类")
        # 标签映射为 ±1（第一类为 -1，第二类为 +1）
        yy = np.where(y == self.classes_[0], -1.0, 1.0)
        m = X.shape[0]
        self._gamma_ = self._resolve_gamma(X)
        K = self._kernel_matrix(X, X)

        alpha = np.zeros(m)
        b = 0.0
        self.n_iter_ = 0
        K_alpha = K @ (alpha * yy)                        # 增量维护 (Kαy)

        for it in range(self.max_iter):
            # 对偶目标函数梯度 g = y⊙(Kαy) − 1；KKT 违反量用 −y⊙g = y − Kαy 衡量
            g = yy * K_alpha - 1.0
            viol = yy - K_alpha
            up = ((yy > 0) & (alpha < self.C)) | ((yy < 0) & (alpha > 0))
            low = ((yy < 0) & (alpha < self.C)) | ((yy > 0) & (alpha > 0))
            if not up.any() or not low.any():
                break
            i = int(np.argmax(np.where(up, viol, -np.inf)))
            j = int(np.argmin(np.where(low, viol, np.inf)))
            if viol[i] - viol[j] < self.tol:              # KKT 条件满足，收敛
                self.n_iter_ = it
                break

            F = K_alpha + b
            Ei, Ej = F[i] - yy[i], F[j] - yy[j]
            ai_old, aj_old = alpha[i], alpha[j]
            if yy[i] != yy[j]:
                L = max(0.0, aj_old - ai_old)
                H = min(self.C, self.C + aj_old - ai_old)
            else:
                L = max(0.0, ai_old + aj_old - self.C)
                H = min(self.C, ai_old + aj_old)
            if H <= L:
                break
            eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
            if eta >= 0:
                break
            alpha[j] = np.clip(aj_old - yy[j] * (Ei - Ej) / eta, L, H)
            if abs(alpha[j] - aj_old) < 1e-8:
                break
            alpha[i] = ai_old + yy[i] * yy[j] * (aj_old - alpha[j])

            b1 = (b - Ei - yy[i] * (alpha[i] - ai_old) * K[i, i]
                  - yy[j] * (alpha[j] - aj_old) * K[i, j])
            b2 = (b - Ej - yy[i] * (alpha[i] - ai_old) * K[i, j]
                  - yy[j] * (alpha[j] - aj_old) * K[j, j])
            if 0 < alpha[i] < self.C:
                b = b1
            elif 0 < alpha[j] < self.C:
                b = b2
            else:
                b = 0.5 * (b1 + b2)

            # 增量更新 Kαy
            K_alpha += ((alpha[i] - ai_old) * yy[i] * K[:, i]
                        + (alpha[j] - aj_old) * yy[j] * K[:, j])
        else:
            self.n_iter_ = self.max_iter

        self.support_ = np.where(alpha > 1e-8)[0]
        self.support_vectors_ = X[self.support_]
        self.dual_coef_ = (alpha * yy)[self.support_]
        self.intercept_ = np.array([b])
        self.n_support_ = np.array([np.sum(yy[self.support_] < 0),
                                    np.sum(yy[self.support_] > 0)])
        self._X_fit = X
        self._yy = yy
        self._alpha = alpha
        if self.kernel == "linear":                       # 线性核可直接给出 w
            self.coef_ = (alpha * yy) @ X
        return self

    # ------------------------------------------------------------------
    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        K = self._kernel_matrix(X, self.support_vectors_)
        return K @ self.dual_coef_ + self.intercept_[0]

    def predict(self, X):
        z = self.decision_function(X)
        return np.where(z >= 0, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    def get_params(self, deep=True):
        return {"C": self.C, "kernel": self.kernel, "gamma": self.gamma,
                "degree": self.degree, "coef0": self.coef0,
                "tol": self.tol, "max_iter": self.max_iter}
