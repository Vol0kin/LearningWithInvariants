import numpy as np
import numpy.typing as npt

from sklearn.base import BaseEstimator, ClassifierMixin

from .utils import generate_encoding, generate_decoding

from ..types import InvariantTypes

class SVMRandomInvariantsECOC(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        encoding: npt.NDArray[np.float64],
        C=1,
        delta=1e-3,
        kernel='rbf',
        gamma='auto',
        invariant_type=InvariantTypes.PROJECTION,
        num_invariants=5,
        num_gen_invariants=20,
        tolerance=100,
        use_v_matrix=False,
        normalize_projections=False,
        verbose=False,
        random_state=None
    ):
        self.encoding = encoding
        self.model_params = {
            'C' : C,
            'delta' : delta,
            'kernel' : kernel,
            'gamma' : gamma,
            'invariant_type' : invariant_type,
            'num_invariants' : num_invariants,
            'num_gen_invariants' : num_gen_invariants,
            'tolerance' : tolerance,
            'use_v_matrix' : use_v_matrix,
            'normalize_projections' : normalize_projections,
            'verbose' : verbose,
            'random_state' : random_state,
        }


    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        self.models = []

        encodings = generate_encoding(self.encoding, y)

        for encoded_y in encodings:
            model = SVMRandomInvariants(**self.model_params)

            model.fit(X, encoded_y)
            self.models.append(model)

        return self


    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        probabilites = np.column_stack([
            model.predict_proba(X)
            for model in self.models
        ])

        prediction = generate_decoding(self.encoding, probabilites)

        return prediction
