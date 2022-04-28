import numpy as np

def positive_class(**kwargs):
    y = kwargs['y']

    return np.ones(len(y))


def mean_in_dimension(dim):
    def inner_mean_in_dimension(**kwargs):
        X = kwargs['X']

        return X[:, dim]
    
    return inner_mean_in_dimension


def _get_positive_class_projection(projected_data, y):
    created_region = False
    positive_class_idx = np.where(y == 1)[0]

    while not created_region:
        center_idx = np.random.choice(positive_class_idx)
        center_point = projected_data[center_idx]
        radius_factor = np.random.uniform(0.15, 0.25)

        min_dim = np.min(projected_data)
        max_dim = np.max(projected_data)
        range_dim = max_dim - min_dim

        radius = range_dim * radius_factor
        min_range = center_point - radius
        max_range = center_point + radius

        points_in_region = np.where(
            (projected_data >= min_range) & (projected_data <= max_range),
            1.0,
            0.0
        )

        points_in_region = np.logical_and(points_in_region, y).astype(float)

        created_region = len(np.where(points_in_region > 0)[0]) > 1
    
    return points_in_region


def random_projection(X, y, only_positives=False):
    d = X.shape[1]

    mean = np.zeros(d)
    cov = np.eye(d)
    projection_vector = np.random.multivariate_normal(mean, cov)

    projected_data = np.dot(X, projection_vector)

    if not only_positives:
        return projected_data

    return _get_positive_class_projection(projected_data, y)


def box(**kwargs):
    X = kwargs['X']
    box_limit = 1.5

    inside_box = np.logical_and(X[:, [1, 5]] >= -box_limit, X[:, [1, 5]] <= box_limit)

    return np.where(inside_box[:, 0] == inside_box[:, 1], 1, 0)


def random_box(X, y, only_positives=False):
    d = X.shape[1]
    n = X.shape[0]
    created_region = False
    positive_class_idx = np.where(y == 1)[0]

    while not created_region:
        center_idx = np.random.choice(positive_class_idx)
        center_point = X[center_idx]
        points_in_region = np.zeros(n)
        radius_factor = np.random.uniform(0.15, 0.25)

        for i in range(d):
            min_dim = np.min(X[:, i])
            max_dim = np.max(X[:, i])
            range_dim = max_dim - min_dim

            radius = range_dim * radius_factor
            min_range = center_point[i] - radius
            max_range = center_point[i] + radius

            idx_in_range = np.where((X[:, i] >= min_range) & (X[:, i] <= max_range))
            points_in_region[idx_in_range] += 1

        points_in_region = points_in_region / d
        points_in_region = points_in_region.astype(int)

        if only_positives:
            points_in_region = np.logical_and(points_in_region, y).astype(float)

        created_region = len(np.where(points_in_region > 0)[0]) > 1

    return points_in_region


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
