import pytest

import numpy as np
from lusi.utils import generate_encoded_labels

@pytest.fixture
def labels():
    np.random.seed(47)

    labels = np.random.choice(5, 1000)

    return labels


@pytest.fixture
def encoding():
    np.random.seed(47)

    encoding = np.random.choice(2, 25).reshape(5, -1).astype(float)

    return encoding


def test_generate_encoded_labels(encoding, labels):
    encoded_problems = generate_encoded_labels(encoding, labels)

    assert np.all(encoded_problems[0, labels == 0] == encoding[0, 0])
    assert np.all(encoded_problems[2, labels == 2] == encoding[2, 2])
    assert np.all(encoded_problems[0, labels == 0] != encoding[3, 0])
