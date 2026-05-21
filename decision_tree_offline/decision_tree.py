import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris, load_wine

class TreeNode:
    def __init__(self, gini, samples, class_counts):
        self.gini = gini
        self.samples = samples
        self.class_counts = class_counts  # dict {class: count}
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None

def calculate_gini(labels):  # label is the target column
    gini = 1.0
    total_count = len(labels)
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    for count in class_counts.values():
        p = count/total_count
        gini -= p*p
    return gini

def split_on_threshold(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold
    return (X[left_mask], y[left_mask], X[right_mask], y[right_mask])

def get_candidate_thresholds(X, feature_index):
    values = np.unique(X[:, feature_index])
    return (values[:-1] + values[1:]) / 2

def find_best_split(X, y, is_random_split, rng):
    n_samples, n_features = X.shape
    best_gini = float("inf")
    best_feature, best_threshold = None, None

    for feature_index in range(n_features):
        values = X[:, feature_index]
        if is_random_split:  # Extra Trees style
            min_val, max_val = values.min(), values.max()
            if min_val == max_val:
                continue
            thresholds = [rng.uniform(min_val, max_val)]
        else:
            thresholds = get_candidate_thresholds(X, feature_index)

        for threshold in thresholds:
            X_l, y_l, X_r, y_r = split_on_threshold(X, y, feature_index, threshold)
            if len(y_l) == 0 or len(y_r) == 0:
                continue

            gini = (len(y_l) * calculate_gini(y_l) + len(y_r) * calculate_gini(y_r)) / n_samples
            if gini < best_gini:
                best_gini = gini
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold

def build_decision_tree(X, y, depth, max_tree_depth, is_random_split,rng):
    class_counts = dict(zip(*np.unique(y, return_counts=True)))
    node = TreeNode(
        gini=calculate_gini(y),
        samples=len(y),
        class_counts=class_counts
    )

    # stopping conditions
    if depth == max_tree_depth or node.gini == 0 or len(y) < 2:
        return node

    feature, threshold = find_best_split(X, y, is_random_split, rng)
    if feature is None:
        return node

    X_l, y_l, X_r, y_r = split_on_threshold(X, y, feature, threshold)

    node.feature_index = feature
    node.threshold = threshold
    node.left = build_decision_tree(X_l, y_l, depth + 1, max_tree_depth, is_random_split, rng)
    node.right = build_decision_tree(X_r, y_r, depth + 1, max_tree_depth, is_random_split, rng)
    return node

def predict_one(x, node):
    if node.left is None and node.right is None:
        return max(node.class_counts, key=node.class_counts.get)

    if x[node.feature_index] <= node.threshold:
        return predict_one(x, node.left)
    else:
        return predict_one(x, node.right)

def predict(X, root):
    return np.array([predict_one(x, root) for x in X])

def predict_proba_one(x, node, n_classes):
    if node.left is None:
        probs = np.zeros(n_classes)
        total = sum(node.class_counts.values())
        for cls, cnt in node.class_counts.items():
            probs[cls] = cnt / total
        return probs

    if x[node.feature_index] <= node.threshold:
        return predict_proba_one(x, node.left, n_classes)
    else:
        return predict_proba_one(x, node.right, n_classes)


def predict_proba(X, root, n_classes):
    return np.array([predict_proba_one(x, root, n_classes) for x in X])

