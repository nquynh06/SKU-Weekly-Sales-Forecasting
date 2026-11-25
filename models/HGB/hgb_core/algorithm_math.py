"""Exact mathematical implementation of Histogram-based Gradient Boosting (HGBDT).

References:
    1. Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
       Annals of Statistics, 29(5), 1189-1232. (Q1)
    2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD. (CORE A*)
    3. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS. (CORE A*)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


def compute_gradients_hessians(y_true: np.ndarray, y_pred: np.ndarray, loss: str = "squared_error") -> Tuple[np.ndarray, np.ndarray]:
    if loss == "squared_error":
        return y_pred - y_true, np.ones_like(y_true, dtype=np.float64)
    elif loss == "absolute_error":
        return np.sign(y_pred - y_true), np.ones_like(y_true, dtype=np.float64)
    elif loss == "poisson":
        exp_f = np.exp(np.clip(y_pred, -20.0, 20.0))
        return exp_f - y_true, np.maximum(exp_f, 1e-6)
    raise ValueError(f"Unsupported loss: {loss}")


def optimal_leaf_weight(G: float, H: float, l2_reg: float) -> float:
    return float(-G / (H + l2_reg + 1e-12))


def regularized_split_gain(G_L: float, H_L: float, G_R: float, H_R: float, l2_reg: float) -> float:
    G_P, H_P = G_L + G_R, H_L + H_R
    return float(
        0.5 * (
            (G_L * G_L) / (H_L + l2_reg + 1e-12)
            + (G_R * G_R) / (H_R + l2_reg + 1e-12)
            - (G_P * G_P) / (H_P + l2_reg + 1e-12)
        )
    )


@dataclass(eq=False)
class _HistNode:
    sample_indices: np.ndarray
    value: float
    depth: int = 0
    feature_idx: Optional[int] = None
    bin_threshold: Optional[int] = None
    missing_go_left: bool = True
    left: Optional["_HistNode"] = None
    right: Optional["_HistNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class ExactHistogramGBDT:
    """Pure-NumPy exact reference implementation of Histogram GBDT (Leaf-wise + NaN routing)."""

    def __init__(self, n_estimators: int = 30, learning_rate: float = 0.1, max_leaf_nodes: int = 15,
                 min_samples_leaf: int = 10, l2_regularization: float = 1.0, max_bins: int = 255,
                 loss: str = "squared_error"):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_bins = min(max_bins, 255)
        self.loss = loss
        self.bin_edges_: List[np.ndarray] = []
        self.trees_: List[_HistNode] = []
        self.init_prediction_: float = 0.0

    def _fit_binning(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        X_binned = np.empty((n_samples, n_features), dtype=np.int16)
        self.bin_edges_ = []
        for j in range(n_features):
            col = X[:, j]
            valid = col[~np.isnan(col)]
            edges = np.unique(np.quantile(valid, np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1])) if len(valid) else np.array([], dtype=np.float64)
            self.bin_edges_.append(edges)
            binned = np.searchsorted(edges, col, side="right").astype(np.int16)
            binned[np.isnan(col)] = -1
            X_binned[:, j] = binned
        return X_binned

    def _transform_binning(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        X_binned = np.empty((n_samples, n_features), dtype=np.int16)
        for j in range(n_features):
            col = X[:, j]
            binned = np.searchsorted(self.bin_edges_[j], col, side="right").astype(np.int16)
            binned[np.isnan(col)] = -1
            X_binned[:, j] = binned
        return X_binned

    def _find_best_split(self, X_binned: np.ndarray, g: np.ndarray, h: np.ndarray, indices: np.ndarray):
        best_gain, best_feat, best_bin, best_missing_left = 0.0, None, None, True
        n_tot = len(indices)
        if n_tot < 2 * self.min_samples_leaf:
            return 0.0, None, None, True

        g_node, h_node = g[indices], h[indices]
        G_tot, H_tot = float(g_node.sum()), float(h_node.sum())

        for j in range(X_binned.shape[1]):
            bins_j = X_binned[indices, j]
            n_bins = len(self.bin_edges_[j]) + 1
            miss_mask = bins_j == -1
            G_miss = float(g_node[miss_mask].sum()) if miss_mask.any() else 0.0
            H_miss = float(h_node[miss_mask].sum()) if miss_mask.any() else 0.0
            n_miss = int(miss_mask.sum())
            valid_mask = ~miss_mask
            if not valid_mask.any():
                continue
            v_bins = bins_j[valid_mask]
            G_cum = np.cumsum(np.bincount(v_bins, weights=g_node[valid_mask], minlength=n_bins))
            H_cum = np.cumsum(np.bincount(v_bins, weights=h_node[valid_mask], minlength=n_bins))
            C_cum = np.cumsum(np.bincount(v_bins, minlength=n_bins))

            for b in range(n_bins - 1):
                for miss_left in (True, False):
                    n_L = int(C_cum[b]) + (n_miss if miss_left else 0)
                    n_R = n_tot - n_L
                    if n_L < self.min_samples_leaf or n_R < self.min_samples_leaf:
                        continue
                    G_L = float(G_cum[b]) + (G_miss if miss_left else 0.0)
                    H_L = float(H_cum[b]) + (H_miss if miss_left else 0.0)
                    gain = regularized_split_gain(G_L, H_L, G_tot - G_L, H_tot - H_L, self.l2_regularization)
                    if gain > best_gain:
                        best_gain, best_feat, best_bin, best_missing_left = gain, j, b, miss_left
        return best_gain, best_feat, best_bin, best_missing_left

    def _fit_one_tree(self, X_binned: np.ndarray, g: np.ndarray, h: np.ndarray) -> _HistNode:
        all_idx = np.arange(X_binned.shape[0])
        root = _HistNode(sample_indices=all_idx, value=optimal_leaf_weight(float(g.sum()), float(h.sum()), self.l2_regularization))
        leaves: List[_HistNode] = [root]
        while len(leaves) < self.max_leaf_nodes:
            best_overall = (0.0, None, None, None, True)
            for leaf in leaves:
                gain, feat, b_thr, m_left = self._find_best_split(X_binned, g, h, leaf.sample_indices)
                if gain > best_overall[0]:
                    best_overall = (gain, leaf, feat, b_thr, m_left)
            best_gain, target_leaf, feat, b_thr, m_left = best_overall
            if best_gain <= 1e-12 or target_leaf is None or feat is None or b_thr is None:
                break
            idx = target_leaf.sample_indices
            col = X_binned[idx, feat]
            left_mask = ((col <= b_thr) | (col == -1)) if m_left else ((col <= b_thr) & (col != -1))
            left_idx, right_idx = idx[left_mask], idx[~left_mask]
            target_leaf.feature_idx, target_leaf.bin_threshold, target_leaf.missing_go_left = feat, b_thr, m_left
            target_leaf.left = _HistNode(left_idx, optimal_leaf_weight(float(g[left_idx].sum()), float(h[left_idx].sum()), self.l2_regularization), target_leaf.depth + 1)
            target_leaf.right = _HistNode(right_idx, optimal_leaf_weight(float(g[right_idx].sum()), float(h[right_idx].sum()), self.l2_regularization), target_leaf.depth + 1)
            leaves.remove(target_leaf)
            leaves.extend([target_leaf.left, target_leaf.right])
        return root

    def _predict_tree(self, node: _HistNode, X_binned: np.ndarray) -> np.ndarray:
        preds = np.empty(X_binned.shape[0], dtype=np.float64)
        def _traverse(curr: _HistNode, mask: np.ndarray) -> None:
            if not mask.any():
                return
            if curr.is_leaf:
                preds[mask] = curr.value
                return
            col = X_binned[:, curr.feature_idx]
            go_left = ((col <= curr.bin_threshold) | (col == -1)) if curr.missing_go_left else ((col <= curr.bin_threshold) & (col != -1))
            _traverse(curr.left, mask & go_left)
            _traverse(curr.right, mask & (~go_left))
        _traverse(node, np.ones(X_binned.shape[0], dtype=bool))
        return preds

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExactHistogramGBDT":
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        X_binned = self._fit_binning(X)
        if self.loss == "absolute_error":
            self.init_prediction_ = float(np.median(y))
        elif self.loss == "poisson":
            self.init_prediction_ = float(np.log(max(float(np.mean(y)), 1e-6)))
        else:
            self.init_prediction_ = float(np.mean(y))
        F = np.full(len(y), self.init_prediction_, dtype=np.float64)
        self.trees_ = []
        for _ in range(self.n_estimators):
            g, h = compute_gradients_hessians(y, F, loss=self.loss)
            tree = self._fit_one_tree(X_binned, g, h)
            self.trees_.append(tree)
            F = F + self.learning_rate * self._predict_tree(tree, X_binned)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_binned = self._transform_binning(np.asarray(X, dtype=np.float64))
        F = np.full(X_binned.shape[0], self.init_prediction_, dtype=np.float64)
        for tree in self.trees_:
            F = F + self.learning_rate * self._predict_tree(tree, X_binned)
        return np.exp(F) if self.loss == "poisson" else F
