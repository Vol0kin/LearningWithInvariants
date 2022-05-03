import numpy as np
import numpy.typing as npt

def generate_encoded_labels(
    encoding: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    num_problems = encoding.shape[1]

    encoded_y = np.array([
        [encoding[label, problem] for label in y]
        for problem in range(num_problems)
    ])

    return encoded_y
