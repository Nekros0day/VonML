import numpy as np


def make_regression_dataset(n_samples=100, noise=0.1, random_state=None):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-1, 1, size=(n_samples, 1))
    true_w = 2.5
    y = true_w * X[:, 0] + rng.normal(scale=noise, size=n_samples)
    return X, y


def make_classification_dataset(n_samples=100, random_state=None):
    rng = np.random.default_rng(random_state)
    centers = np.array([[-1, -1], [1, 1]])
    X = np.vstack([
        rng.normal(loc=centers[0], scale=0.5, size=(n_samples // 2, 2)),
        rng.normal(loc=centers[1], scale=0.5, size=(n_samples - n_samples // 2, 2)),
    ])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
    return X, y.reshape(-1, 1)

