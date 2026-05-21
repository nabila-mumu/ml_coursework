import numpy as np

def bootstrap_sample(X, y, rng, min_samples_split):
    n_samples = X.shape[0]
    size = max(min_samples_split, n_samples)
    indices = rng.choice(n_samples, size, replace=True)
    return X[indices], y[indices]


def subsample_features(X, n_features, rng):
    total_features = X.shape[1]
    feature_indices = rng.choice(
        total_features, n_features, replace=False
    )
    return X[:, feature_indices], feature_indices