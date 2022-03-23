import numpy as np

def positive_class(**kwargs):
    y = kwargs['y']

    return np.ones(len(y))


def mean_in_dimension(dim):
    def inner_mean_in_dimension(**kwargs):
        X = kwargs['X']

        return X[:, dim]
    
    return inner_mean_in_dimension


def random_projection(**kwargs):
    X = kwargs['X']
    
    n_dims = X.shape[1]

    mean = np.zeros(n_dims)
    cov = np.eye(n_dims)
    projection_vector = np.random.multivariate_normal(mean, cov)

    projected_data = np.dot(X, projection_vector)

    return projected_data


def box(**kwargs):
    X = kwargs['X']
    box_limit = 1.5

    inside_box = np.logical_and(X[:, [1, 5]] >= -box_limit, X[:, [1, 5]] <= box_limit)

    return np.where(inside_box[:, 0] == inside_box[:, 1], 1, 0)
