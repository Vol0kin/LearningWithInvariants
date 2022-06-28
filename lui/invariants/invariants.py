import numpy as np

def positive_class(y):
    return np.ones(len(y))


def mean_in_dimension(X, dim):
    return X[:, dim]


def random_projection(X):
    d = X.shape[1]

    mean = np.zeros(d)
    cov = np.eye(d)
    projection_vector = np.random.multivariate_normal(mean, cov)

    projected_data = np.dot(X, projection_vector)

    return projected_data


def random_hyperplane(X):
    n, d = X.shape

    mean = np.zeros(d)
    cov = np.eye(d)
    hyperplane_vector = np.random.multivariate_normal(mean, cov)

    center_idx = np.random.choice(n)
    center = X[center_idx]
    hyperplane_dist = np.dot(X - center, hyperplane_vector)

    # The center point is considered as a positive sample
    labels = np.where(hyperplane_dist < 0, 0, 1)

    return labels
