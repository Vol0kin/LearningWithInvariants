from enum import Enum

import numpy as np
from scipy.spatial import distance
from lusi.invariants import *

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy.typing as npt
from typing import Tuple

class InvariantType(str, Enum):
    PROJECTION = 'PROJECTION'
    HYPERPLANE = 'HYPERPLANE'

class SVMRandomInvariants(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        C=1,
        delta=1e-3,
        kernel='rbf',
        gamma='auto',
        invariant_type=InvariantType.PROJECTION,
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
            random_projection(self.X, self.y)
            for _ in range(self.num_gen_invariants)
        ])

        if self.normalize_projections:
            random_projections = random_projections / np.sqrt(self.d)

        return random_projections


    def _generate_random_hyperplanes(self) -> npt.NDArray[np.float64]:
        random_hyperplanes = np.array([
            random_hyperplane(self.X)
            for _ in range(self.num_gen_invariants)
        ])

        return random_hyperplanes


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
        if self.invariant_type == InvariantType.PROJECTION:
            invariant_generation_func = self._generate_random_projections
        else:
            invariant_generation_func = self._generate_random_hyperplanes
        
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
        invariants = []

        while n_tries < self.tolerance and len(invariants) < self.num_invariants:
            n_tries += 1

            # Generate random projection invariants
            predicates = invariant_generation_func()

            T_values = []

            # Evaluate the random projections
            for pred in predicates:
                num = np.dot(pred, np.dot(K, A)) + c * np.dot(pred, ones) - np.dot(pred, self.y)
                den = np.dot(self.y, pred) + 1
                T_values.append(np.abs(num) / den)
            
            T_max = np.max(T_values)

            if T_max > self.delta:
                if self.verbose:
                    print(f'Selected invariant after {n_tries} tries with T={T_max}')
                    # print(T_values)

                # Update control variables
                n_tries = 0

                invariants.append(predicates[np.argmax(T_values)])
                invariants_arr = np.array(invariants)

                A_s = np.array([np.dot(VK_perturbed_inv, phi) for phi in invariants_arr])

                # Create system of equations
                c_1 = np.dot(ones, np.dot(VK, A_c)) - np.dot(ones, np.dot(V, ones))

                mu_1 = np.array([
                    np.dot(ones, np.dot(VK, phi)) - np.dot(ones, phi)
                    for phi in invariants_arr
                ])

                rh_1 = np.dot(ones, np.dot(VK, A_v)) - np.dot(ones, np.dot(V, self.y))

                c_2 = np.array([
                    np.dot(A_c, np.dot(K, phi)) - np.dot(ones, phi)
                    for phi in invariants
                ])

                mu_2 = np.array([
                    np.array([
                        np.dot(A_s[s], np.dot(K, invariants_arr[k]))
                        for s in range(len(invariants))
                    ])
                    for k in range(len(invariants))
                ])

                rh_2 = np.array([
                    np.dot(A_v, np.dot(K, phi)) - np.dot(self.y, phi)
                    for phi in invariants_arr
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
                        for s in range(len(invariants))
                    ]), axis=0
                )
                A = A_v - c * A_c - sum_mu_As

                if self.verbose:
                    print('Invariants weights: ', mu)

        if self.verbose:
            print('Finished training')
            print(f'Num. invariants: {len(invariants)}\tNum. tries: {n_tries}')
        
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
            self.kernel = rbf_kernel
            K = self.kernel(X, X, gamma=self.gamma)
        elif self.kernel == 'linear':
            self.kernel = linear_kernel
            K = self.kernel(X, X)

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
    

    def _decision_function(self, Z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        preds = []

        for z in Z:
            preds.append(np.dot(self.A, self.kernel(self.X,[z])) + self.c)

        return np.array(preds).flatten()


    def plot_decision_boundary(self, title=''):
        h = .02
        x_min, x_max = self.X[:, 0].min() - .2, self.X[:, 0].max() + .2
        y_min, y_max = self.X[:, 1].min() - .2, self.X[:, 1].max() + .2
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        viz=np.c_[xx.ravel(),yy.ravel()]
        Z = self._decision_function(viz)
        Z = Z.reshape(xx.shape)
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        plt.figure(figsize=(5,5))
        plt.contourf(xx, yy, Z, levels=np.linspace(-1.3,2.3,13), cmap=cm, alpha=.8)
        plt.contour(xx, yy, Z, levels=[0.5], linestyles='dashed')
        plt.scatter(self.X[:,0], self.X[:,1], c=self.y, cmap=cm_bright, edgecolors='k')
        plt.tight_layout()
        plt.title(title)
        plt.show()

class SVMRandomInvariantsECOC(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        encoding: npt.NDArray[np.float64],
        C=1,
        delta=1e-3,
        kernel='rbf',
        gamma='auto',
        invariant_type=InvariantType.PROJECTION,
        num_invariants=5,
        num_gen_invariants=20,
        tolerance=100,
        use_v_matrix=False,
        normalize_projections=False,
        verbose=False,
        random_state=None
    ):
        self.encoding = encoding
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


    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        self.models = []
        num_problems = self.encoding.shape[1]

        for problem in range(num_problems):
            encoded_y = np.array([self.encoding[label, problem] for label in y])

            model = SVMRandomInvariants(
                C = self.C,
                delta = self.delta,
                kernel = self.kernel,
                gamma = self.gamma,
                invariant_type = self.invariant_type,
                num_invariants = self.num_invariants,
                num_gen_invariants = self.num_gen_invariants,
                tolerance = self.tolerance,
                use_v_matrix = self.use_v_matrix,
                normalize_projections = self.normalize_projections,
                verbose = self.verbose,
                random_state = self.random_state
            )

            model.fit(X, encoded_y)
            self.models.append(model)

        return self


    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        probabilites = np.column_stack([
            model.predict_proba(X)
            for model in self.models
        ])

        dists = distance.cdist(self.encoding, probabilites, 'euclidean')
        prediction = np.argmax(dists, axis=0)

        return prediction

