import numpy as np
from lusi import invariants

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy.typing as npt
from typing import Tuple

from ..types import InvariantTypes

class SVMRandomInvariants(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
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
        self.C = C
        self.delta = delta
        self.kernel = kernel
        self.gamma = gamma
        self.invariant_type = invariant_type
        self.num_invariants = num_invariants
        self.num_gen_invariants = num_gen_invariants
        self.tolerance = tolerance
        self.use_v_matrix = use_v_matrix
        self.normalize_projections = normalize_projections
        self.verbose = verbose
        self.random_state = random_state
    

    def _generate_V_matrix(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # max_dims are the upper bounds of each dimension.
        # Since there is not clear specification on which values have to be picked,
        # using the maximum for each dimension should be a good choice
        max_dims = np.max(X, axis=0)

        V = np.array([
            [
                np.prod(max_dims - np.maximum(x_i, x_j))
                for x_j in X
            ] 
            for x_i in X
        ])

        return V


    def _generate_random_projections(self) -> npt.NDArray[np.float64]:
        random_projections = np.array([
            invariants.random_projection(self.X)
            for _ in range(self.num_gen_invariants)
        ])

        if self.normalize_projections:
            random_projections = random_projections / np.sqrt(self.d)

        return random_projections


    def _generate_random_hyperplanes(self) -> npt.NDArray[np.float64]:
        random_hyperplanes = np.array([
            invariants.random_hyperplane(self.X)
            for _ in range(self.num_gen_invariants)
        ])

        return random_hyperplanes


    def _generate_vapnik_invariants(self) -> npt.NDArray[np.float64]:
        vapnik_invariants = [invariants.positive_class(self.y)]
        vapnik_invariants.extend([
            invariants.mean_in_dimension(self.X, dim)
            for dim in range(self.X.shape[1])
        ])

        return vapnik_invariants


    def _generate_all_invariants(self) -> npt.NDArray[np.float64]:
        random_projections = self._generate_random_projections()
        random_hyperplanes = self._generate_random_hyperplanes()
        vapnik_invariants = self._generate_vapnik_invariants()

        all_invariants = np.r_[random_projections, random_hyperplanes, vapnik_invariants]

        return all_invariants


    def _simple_inference(
        self,
        K: npt.NDArray[np.float64],
    ) -> Tuple[npt.NDArray[np.float64], float]:
        # Compute V matrix and Gramm matrix
        V = self._generate_V_matrix(self.X) if self.use_v_matrix else np.eye(self.l)

        # Helpers
        ones_vector = np.ones(self.l)
        perturbation = np.diagflat(self.C * ones_vector)

        VK_prod = np.dot(V, K)
        perturbed_VK_inv = np.linalg.inv(VK_prod + perturbation)

        # Compute A_b and A_c, which will be used to compute c
        A_b = np.dot(perturbed_VK_inv, np.dot(V, self.y))
        A_c = np.dot(perturbed_VK_inv, np.dot(V, ones_vector))

        numerator = np.dot(ones_vector, np.dot(VK_prod, A_b)) - np.dot(ones_vector, np.dot(V, self.y))
        denominator = np.dot(ones_vector, np.dot(VK_prod, A_c)) - np.dot(ones_vector, np.dot(V, ones_vector))

        # Compute closed-form solution of the minimization problem
        c = numerator / denominator
        A = A_b - c * A_c

        if self.verbose:
            print(f'c: {c}')
            print(f'A: {A}')

        return A, c


    def _invariants_inference(self, K: npt.NDArray[np.float64]):
        match self.invariant_type:
            case InvariantTypes.PROJECTION:
                invariant_generation_func = self._generate_random_projections
            case InvariantTypes.HYPERPLANE:
                invariant_generation_func = self._generate_random_hyperplanes
            case InvariantTypes.VAPNIK:
                invariant_generation_func = self._generate_vapnik_invariants
            case InvariantTypes.ALL:
                invariant_generation_func = self._generate_all_invariants
        
        if self.verbose:
            print(f'Using {self.invariant_type} invariant')

        A, c = self._simple_inference(K)

        # Compute V matrix and Gramm matrix
        V = self._generate_V_matrix(self.X) if self.use_v_matrix else np.eye(self.l)

        # Create auxiliar variables
        ones = np.ones(self.l)
        VK = np.dot(V, K)
        VK_perturbed_inv = np.linalg.inv(VK + self.C * np.eye(self.l))

        # Compute vectors
        A_v = np.dot(VK_perturbed_inv, np.dot(V, self.y))
        A_c = np.dot(VK_perturbed_inv, np.dot(V, ones))

        n_tries = 0
        self.invariants = []

        while n_tries < self.tolerance and len(self.invariants) < self.num_invariants:
            n_tries += 1

            # Generate random projection invariants
            predicates = invariant_generation_func()

            T_values = []

            # Evaluate the invariants
            for pred in predicates:
                num = np.dot(pred, np.dot(K, A)) + c * np.dot(pred, ones) - np.dot(pred, self.y)
                den = np.dot(self.y, pred)
                T = np.abs(num) / den if den > 0.0 else 0.0
                T_values.append(T)
            
            T_max = np.max(T_values)
            T_max_idx = np.argmax(T_values)

            if T_max > self.delta:
                if self.verbose:
                    print(f'Selected invariant {T_max_idx} after {n_tries} tries with T={T_max}')

                # Update control variables
                n_tries = 0

                if self.invariant_type != InvariantTypes.ALL:
                    invariant_type = self.invariant_type
                else:
                    if T_max_idx < self.num_gen_invariants:
                        invariant_type = InvariantTypes.PROJECTION
                    elif T_max_idx <= self.num_gen_invariants * 2:
                        invariant_type = InvariantTypes.HYPERPLANE
                    else:
                        invariant_type = InvariantTypes.VAPNIK

                self.invariants.append({'invariant': predicates[T_max_idx], 'type': invariant_type, 'T_value': T_max})

                A_s = np.array([np.dot(VK_perturbed_inv, phi['invariant']) for phi in self.invariants])

                # Create system of equations
                c_1 = np.dot(ones, np.dot(VK, A_c)) - np.dot(ones, np.dot(V, ones))

                mu_1 = np.array([
                    np.dot(ones, np.dot(VK, phi['invariant'])) - np.dot(ones, phi['invariant'])
                    for phi in self.invariants
                ])

                rh_1 = np.dot(ones, np.dot(VK, A_v)) - np.dot(ones, np.dot(V, self.y))

                c_2 = np.array([
                    np.dot(A_c, np.dot(K, phi['invariant'])) - np.dot(ones, phi['invariant'])
                    for phi in self.invariants
                ])

                mu_2 = np.array([
                    np.array([
                        np.dot(A_s[s], np.dot(K, self.invariants[k]['invariant']))
                        for s in range(len(self.invariants))
                    ])
                    for k in range(len(self.invariants))
                ])

                rh_2 = np.array([
                    np.dot(A_v, np.dot(K, phi['invariant'])) - np.dot(self.y, phi['invariant'])
                    for phi in self.invariants
                ])

                a_1 = np.concatenate(([c_1], mu_1))
                a_2 = np.vstack(([c_2], mu_2.T)).T
                a = np.concatenate(([a_1], a_2), axis=0)
                b = np.concatenate(([rh_1], rh_2))

                solution = np.linalg.solve(a, b)

                c, mu = solution[0], solution[1:]

                # The sum can be replaced with a dot product
                sum_mu_As = np.sum(
                    np.array([
                        mu[s] * A_s[s]
                        for s in range(len(self.invariants))
                    ]), axis=0
                )
                A = A_v - c * A_c - sum_mu_As

                if self.verbose:
                    print('Invariants weights: ', mu)

        if self.verbose:
            print('Finished training')
            print(f'Num. invariants: {len(self.invariants)}\tNum. tries: {n_tries}')
        
        return A, c


    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
    ):
        self.X = X
        self.y = y
        self.l, self.d = X.shape

        np.random.seed(self.random_state)

        if self.gamma == 'auto':
            self.gamma = 1 / self.d
        
        if self.kernel == 'rbf':
            self.kernel_func = rbf_kernel
            K = self.kernel_func(X, X, gamma=self.gamma)
        elif self.kernel == 'linear':
            self.kernel_func = linear_kernel
            K = self.kernel_func(X, X)

        if self.num_invariants == 0:
            self.A, self.c = self._simple_inference(K)
        else:
            self.A, self.c = self._invariants_inference(K)

        return self


    def predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        probabilites = np.dot(self.A, rbf_kernel(self.X, X, gamma=self.gamma)) + self.c

        probabilites = np.clip(probabilites, 0, 1)

        return probabilites


    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Apply decision rule
        probabilites = self.predict_proba(X)

        # Get label: if the result of the decision rule is smaller than 0.5, then
        # it's classified as 0. Otherwise, it's classified as 1
        prediction = np.where(probabilites < 0.5, 0, 1)

        return prediction
