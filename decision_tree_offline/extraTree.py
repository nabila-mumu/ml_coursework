import numpy as np
from helper import bootstrap_sample, subsample_features
from decision_tree import build_decision_tree, predict, predict_proba

def extra_trees_train(X_train, y_train, n_trees, max_tree_depth, n_features, random_state=42):
    rng = np.random.RandomState(random_state)
    forest = []

    for i in range(n_trees):
        # use whole training data for extra trees
        X_sub, feature_idx = subsample_features(X_train, n_features, rng)

        tree = build_decision_tree(X_sub, y_train, depth=0, max_tree_depth=max_tree_depth, is_random_split=True, rng=rng)

        forest.append((tree, feature_idx))

    return forest

def extra_trees_predict(X, forest):
    predictions = []

    for tree, feature_idx in forest:
        preds = predict(X[:, feature_idx], tree)
        predictions.append(preds)

    predictions = np.array(predictions)

    return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

def extra_trees_predict_proba(X, forest, n_classes):
    probs = []

    for tree, feature_idx in forest:
        p = predict_proba(X[:, feature_idx], tree, n_classes)
        probs.append(p)

    return np.mean(probs, axis=0)

