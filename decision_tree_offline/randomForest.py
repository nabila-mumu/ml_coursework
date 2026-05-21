import numpy as np
from helper import bootstrap_sample, subsample_features
from decision_tree import build_decision_tree, predict, predict_proba

def random_forest_train(X_train, y_train, n_trees, max_tree_depth, min_samples_split, n_features, random_state=42):
    rng = np.random.RandomState(random_state)
    forest = []

    for i in range(n_trees):
        X_boot, y_boot = bootstrap_sample(X_train, y_train, rng, min_samples_split=min_samples_split)  # Bootstrap data
        X_sub, feature_idx = subsample_features(X_boot, n_features, rng)  # Feature subsampling
        # Train tree
        tree = build_decision_tree(X_sub, y_boot, depth=0, max_tree_depth=max_tree_depth, is_random_split=False, rng=rng)
        forest.append((tree, feature_idx))
    return forest

def random_forest_predict(X, forest):
    predictions = []
    for tree, feature_idx in forest:
        preds = predict(X[:, feature_idx], tree)
        predictions.append(preds)

    predictions = np.array(predictions)

    # Majority vote
    return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

def random_forest_predict_proba(X, forest, n_classes):
    probs = []

    for tree, feature_idx in forest:
        p = predict_proba(X[:, feature_idx], tree, n_classes)
        probs.append(p)

    return np.mean(probs, axis=0)

