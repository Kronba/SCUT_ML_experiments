import numpy as np


def _relu(z):
    return np.maximum(z, 0.0)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLPClassifier:
    """单隐藏层 BP 神经网络（反向传播 + Adam）。

    参数
    ----
    hidden_layer_sizes : tuple
        隐藏层神经元个数，例如 (2,) 表示一个含 2 个神经元的隐藏层。
    activation : {"relu"}
        激活函数。
    learning_rate_init : float
        Adam 的初始学习率。
    alpha : float
        L2 正则化系数。
    max_iter : int
        最大迭代轮数（每轮遍历一次训练集）。
    batch_size : int or "auto"
        小批量大小；"auto" 取 min(200, 样本数)，与 sklearn 一致。
    tol : float
        训练损失连续若干轮下降小于该值时提前停止。
    n_iter_no_change : int
        提前停止的耐心轮数。
    random_state : int or None
        随机种子，用于参数初始化与每轮打乱样本。
    """

    def __init__(self, hidden_layer_sizes=(2,), activation="relu",
                 learning_rate_init=0.001, alpha=1e-4, max_iter=1000,
                 batch_size="auto", tol=1e-4, n_iter_no_change=10,
                 random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate_init = learning_rate_init
        self.alpha = alpha
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

    # ------------------------------------------------------------------
    def _init_params(self, n_in, n_hidden, n_out, rng):
        # Glorot 均匀初始化，与 sklearn 的默认初始化一致
        lim1 = np.sqrt(6.0 / (n_in + n_hidden))
        lim2 = np.sqrt(6.0 / (n_hidden + n_out))
        W1 = rng.uniform(-lim1, lim1, (n_in, n_hidden))
        b1 = np.zeros(n_hidden)
        W2 = rng.uniform(-lim2, lim2, (n_hidden, n_out))
        b2 = np.zeros(n_out)
        return [W1, b1, W2, b2]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        n_samples, n_in = X.shape
        n_hidden = int(self.hidden_layer_sizes[0])
        n_out = self.classes_.size

        # 标签转 one-hot
        Y = np.zeros((n_samples, n_out))
        Y[np.arange(n_samples), np.searchsorted(self.classes_, y)] = 1.0

        rng = np.random.default_rng(self.random_state)
        params = self._init_params(n_in, n_hidden, n_out, rng)
        W1, b1, W2, b2 = params

        batch = (min(200, n_samples) if self.batch_size == "auto"
                 else int(self.batch_size))
        lr = self.learning_rate_init
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        step = 0
        self.loss_curve_ = []
        best_loss, no_improve = np.inf, 0

        for epoch in range(self.max_iter):
            order = rng.permutation(n_samples)
            for start in range(0, n_samples, batch):
                idx = order[start:start + batch]
                xb, yb = X[idx], Y[idx]

                # ---- 前向传播
                z1 = xb @ W1 + b1
                a1 = _relu(z1)
                z2 = a1 @ W2 + b2
                p = _softmax(z2)

                # ---- 反向传播（交叉熵 + L2）
                dz2 = (p - yb) / len(idx)
                dW2 = a1.T @ dz2 + self.alpha * W2
                db2 = dz2.sum(axis=0)
                da1 = dz2 @ W2.T
                dz1 = da1 * (z1 > 0)
                dW1 = xb.T @ dz1 + self.alpha * W1
                db1 = dz1.sum(axis=0)

                grads = [dW1, db1, dW2, db2]
                step += 1
                for k, g in enumerate(grads):
                    m[k] = beta1 * m[k] + (1 - beta1) * g
                    v[k] = beta2 * v[k] + (1 - beta2) * (g ** 2)
                    m_hat = m[k] / (1 - beta1 ** step)
                    v_hat = v[k] / (1 - beta2 ** step)
                    params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)
                W1, b1, W2, b2 = params

            # 每轮记录一次训练损失
            p_all = _softmax(_relu(X @ W1 + b1) @ W2 + b2)
            loss = -np.mean(np.sum(Y * np.log(p_all + 1e-12), axis=1))
            loss += 0.5 * self.alpha * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
            self.loss_curve_.append(float(loss))
            self.n_iter_ = epoch + 1

            if best_loss - loss < self.tol:
                no_improve += 1
                if no_improve >= self.n_iter_no_change:
                    break
            else:
                no_improve = 0
            best_loss = min(best_loss, loss)

        self.coefs_ = [W1, W2]
        self.intercepts_ = [b1, b2]
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        W1, b1, W2, b2 = self.coefs_[0], self.intercepts_[0], self.coefs_[1], self.intercepts_[1]
        return _softmax(_relu(X @ W1 + b1) @ W2 + b2)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    def get_params(self, deep=True):
        return {"hidden_layer_sizes": self.hidden_layer_sizes,
                "activation": self.activation,
                "learning_rate_init": self.learning_rate_init,
                "alpha": self.alpha, "max_iter": self.max_iter,
                "batch_size": self.batch_size, "tol": self.tol,
                "n_iter_no_change": self.n_iter_no_change,
                "random_state": self.random_state}
