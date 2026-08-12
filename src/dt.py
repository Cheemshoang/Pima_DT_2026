import pandas as pd
import numpy as np
import sklearn 
from collections import Counter
from pathlib import Path

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        # Decision node parameters
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
        
    def is_leaf_node(self):
        return self.value is not None
    
    
    
class DecisionTree:
    def __init__(self, criterion="gini", min_samples_split=3, max_depth=5):
        self.criterion = criterion 
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        #key
        X = X.values.astype(float)
        y = y.values.astype(int)
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        #might be this is the way
        feat_idxs = np.random.choice(n_features, n_features, replace=False)

        # Find the best split based on the chosen criterion
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        
        # If no split improves the gain, make it a leaf
        if best_feat is None or best_thresh is None:
             leaf_value = self._most_common_label(y)
             return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._calculate_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _calculate_gain(self, y, X_column, split_thresh):
        parent_impurity = self._impurity(y)

        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        imp_l, imp_r = self._impurity(y[left_idxs]), self._impurity(y[right_idxs])
        child_impurity = (n_l / n) * imp_l + (n_r / n) * imp_r

        return parent_impurity - child_impurity

    def _impurity(self, y):
        if self.criterion == "entropy":
            return self._entropy(y)
        else:
            return self._gini(y)

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _gini(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return 1 - np.sum(ps ** 2)

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        counter = Counter(y)
        if len(counter) == 0:
            return None
        return counter.most_common(1)[0][0]

    def _traverse_tree(self, x, node):
        #backtrack
        if node.is_leaf_node():
            return node.value
        
        if node.threshold is None:
            return None
        if x[node.feature] <= float(node.threshold):
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

