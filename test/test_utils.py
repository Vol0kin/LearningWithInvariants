import pytest

import numpy as np
from lusi.utils import generate_decoding, generate_encoding

@pytest.fixture
def labels():
    np.random.seed(47)

    labels = np.random.choice(5, 1000)

    return labels


@pytest.fixture
def encoding():
    # One vs Rest encoding
    return np.eye(5, dtype=int)


@pytest.fixture
def probabilites():
    return np.array([
        [0., 0., 1., 0., 0.],
        [0.6, 0.1, 0.25, 0.8, 0.3]
    ])


def test_generate_encoding(encoding, labels):
    encodings = generate_encoding(encoding, labels)

    assert np.all(encodings[0, labels == 0] == encoding[0, 0])
    assert np.all(encodings[0, labels != 0] != encoding[0, 0])

    assert np.all(encodings[2, labels == 2] == encoding[2, 2])


def test_generate_decoding(encoding, probabilites):
    predictions = generate_decoding(encoding, probabilites)

    assert predictions[0] == 2
    assert predictions[1] == 3
