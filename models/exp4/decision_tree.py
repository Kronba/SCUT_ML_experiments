import numpy as np


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "proba", "n_samples", "value")

    def __init__(self):
        self.feature = -1
        self.threshold = 0.0
        self.left = None
        self.right = None
        self.proba = None
        self.n_samples = 0
        self.value = 0


class DecisionTreeClassifier:
    """决策树分类器。

    参数
    ----
    criterion : {"entropy", "gini"}
        划分准则。
    max_depth : int or None
        最大深度。
    min_samples_split : int
        内部节点继续划分所需的最少样本数。
    min_samples_leaf : int
        叶节点允许的最少样本数。
    class_weight : None or {"balanced"} or dict
        类别权重；"balanced" 按 N/(K·N_k) 自动赋权。
    random_state : int or None
        预留参数，保证接口一致。
    """

    def __init__(self, criterion="entropy", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, class_weight=None, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state

    # ------------------------------------------------------------------
    def _impurity(self, counts):
        """counts: (..., K) 的加权类别计数，返回 (...,) 的不纯度。"""
        totals = counts.sum(axis=-1, keepdims=True)
        p = counts / np.maximum(totals, 1e-12)
        if self.criterion == "entropy":
            return -np.sum(np.where(p > 0, p * np.log2(np.maximum(p, 1e-12)), 0.0), axis=-1)
        return 1.0 - np.sum(p ** 2, axis=-1)

    def _best_split(self, X, y, w, idx):
        n_features = X.shape[1]
        K = self.classes_.size
        total_counts = np.array([w[idx][y[idx] == k].sum() for k in range(K)])
        total_w = total_counts.sum()
        parent = float(self._impurity(total_counts))
        best = (0.0, None, None)

        for f in range(n_features):
            order = np.argsort(X[idx, f], kind="mergesort")
            xs = X[idx[order], f]
            ys = y[idx[order]]
            ws = w[idx[order]]
            counts = np.zeros((len(xs), K))
            counts[np.arange(len(xs)), ys] = ws
            cum = np.cumsum(counts, axis=0)
            valid = np.where(xs[:-1] < xs[1:])[0]
            if valid.size == 0:
                continue
            left = cum[valid]
            right = cum[-1] - left
            n_left = left.sum(axis=1)
            n_right = right.sum(axis=1)
            # 满足叶节点最小样本数约束
            ok = (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
            if not ok.any():
                continue
            imp = (self._impurity(left) * n_left + self._impurity(right) * n_right) / total_w
            gain = parent - imp
            gain = np.where(ok, gain, -np.inf)
            k = int(np.argmax(gain))
            if gain[k] > best[0]:
                thr = 0.5 * (xs[valid[k]] + xs[valid[k] + 1])
                best = (float(gain[k]), f, thr)
        return best

    def _build(self, X, y, w, idx, depth):
        node = _Node()
        node.n_samples = len(idx)
        counts = np.array([w[idx][y[idx] == k].sum() for k in range(self.classes_.size)])
        node.proba = counts / max(counts.sum(), 1e-12)
        node.value = int(np.argmax(node.proba))

        if (self.max_depth is not None and depth >= self.max_depth) or \
                len(idx) < self.min_samples_split or node.proba.max() >= 1.0 - 1e-12:
            return node
        gain, f, thr = self._best_split(X, y, w, idx)
        if f is None or gain <= 1e-12:
            return node

        mask = X[idx, f] <= thr
        if mask.sum() == 0 or (~mask).sum() == 0:
            return node
        node.feature, node.threshold = f, thr
        left_idx, right_idx = idx[mask], idx[~mask]
        if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
            return node
        node.left = self._build(X, y, w, left_idx, depth + 1)
        node.right = self._build(X, y, w, right_idx, depth + 1)
        return node

    def _accumulate_importance(self):
        self.feature_importances_ = np.zeros(self.n_features_in_)
        total = np.zeros(self.n_features_in_)

        def walk(node):
            if node.left is None:
                return
            total[node.feature] += node.n_samples
            walk(node.left)
            walk(node.right)

        walk(self.tree_)
        if total.sum() > 0:
            self.feature_importances_ = total / total.sum()

    # ------------------------------------------------------------------
    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        y_idx = np.searchsorted(self.classes_, y)

        w = (np.ones(X.shape[0]) if sample_weight is None
             else np.asarray(sample_weight, dtype=float).ravel().copy())

        # 类别权重
        if self.class_weight == "balanced":
            for k, c in enumerate(self.classes_):
                n_k = np.sum(y_idx == k)
                if n_k > 0:
                    w[y_idx == k] *= X.shape[0] / (self.classes_.size * n_k)
        elif isinstance(self.class_weight, dict):
            for k, c in enumerate(self.classes_):
                if c in self.class_weight:
                    w[y_idx == k] *= self.class_weight[c]

        self.tree_ = self._build(X, y_idx, w, np.arange(X.shape[0]), 0)
        self._accumulate_importance()
        return self

    def _predict_one(self, x):
        node = self.tree_
        while node.left is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        out = np.zeros((X.shape[0], self.classes_.size))
        for i, x in enumerate(X):
            out[i] = self._predict_one(x).proba
        return out

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self.classes_[np.array([self._predict_one(x).value for x in X])]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    def get_params(self, deep=True):
        return {"criterion": self.criterion, "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "class_weight": self.class_weight,
                "random_state": self.random_state}
