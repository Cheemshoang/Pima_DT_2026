import pandas as pd
import numpy as np
import sklearn 
from collections import Counter
from pathlib import Path

class XGBoostNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, weight=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.weight = weight  
        
    def is_leaf_node(self):
        return self.weight is not None

class XGBoostTree:
    def __init__(self, max_depth=3, min_samples_split=2, lambda_reg=1.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lambda_reg = lambda_reg  # The Regularization penalty
        self.root = None

    def fit(self, X, g, h):
        # Gradients (g) and Hessians (h) instead of y
        self.root = self._grow_tree(X, g, h)

    def _grow_tree(self, X, g, h, depth=0):
        n_samples, n_features = X.shape
        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            leaf_weight = self._calculate_leaf_weight(g, h)
            return XGBoostNode(weight=leaf_weight)

        best_feat, best_thresh = self._best_split(X, g, h, n_features)
        
        if best_feat is None:
             leaf_weight = self._calculate_leaf_weight(g, h)
             return XGBoostNode(weight=leaf_weight)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        
        left = self._grow_tree(X[left_idxs, :], g[left_idxs], h[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], g[right_idxs], h[right_idxs], depth + 1)
        
        return XGBoostNode(best_feat, best_thresh, left, right)

    def _best_split(self, X, g, h, n_features):
        best_gain = -1
        split_idx, split_thresh = None, None
        
        # Calculate 
        root_similarity = self._similarity_score(g, h)

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                left_idxs, right_idxs = self._split(X_column, threshold)
                
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue
                    
                left_sim = self._similarity_score(g[left_idxs], h[left_idxs])
                right_sim = self._similarity_score(g[right_idxs], h[right_idxs])
                
                # Gain: L + R - root
                gain = left_sim + right_sim - root_similarity
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _similarity_score(self, g, h):
        # hessian + lambda
        return (np.sum(g) ** 2) / (np.sum(h) + self.lambda_reg)

    def _calculate_leaf_weight(self, g, h):

        return -np.sum(g) / (np.sum(h) + self.lambda_reg)

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.weight
        if float(x[node.feature]) <= float(node.threshold):
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    
    
class XGBoostModel:
    def __init__(self, n_trees=10, learning_rate=0.1, max_depth=3, lambda_reg=1.0):
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.lambda_reg = lambda_reg
        self.trees = []
        self.base_score = None

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
  
        self.base_score = np.mean(y)
        predictions = np.full(y.shape, self.base_score)
        
        for i in range(self.n_trees):
            g = predictions - y           # Gradient 
            h = np.ones_like(y)           # Hessian
            
            # Residual learning
            tree = XGBoostTree(max_depth=self.max_depth, lambda_reg=self.lambda_reg)
            tree.fit(X, g, h)
            self.trees.append(tree)
            predictions += self.learning_rate * tree.predict(X)

    def predict(self, X):
        X = np.array(X, dtype=float)
        # Start with the base guess
        preds = np.full(X.shape[0], self.base_score)
        # Pass the data through every tree in the sequence and add their weighted fixes
        for tree in self.trees:
            preds += self.learning_rate * tree.predict(X)
        return np.round(preds).astype(int)