import numpy as np
import numpy.typing as npt

from scipy.spatial import distance

def generate_encoding(
    encoding: npt.NDArray[np.int32],
    y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    num_problems = encoding.shape[1]

    encoding = np.array([
        [encoding[label, problem] for label in y]
        for problem in range(num_problems)
    ])

    return encoding


def generate_decoding(
    encoding: npt.NDArray[np.int32],
    probabilites: npt.NDArray[np.float64],
    metric='euclidean'
) -> npt.NDArray[np.int32]:
    distances = distance.cdist(encoding, probabilites, metric=metric)
    decoding = np.argmin(distances, axis=0)

    return decoding
