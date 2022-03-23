import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

import numpy.typing as npt
from typing import Callable, List, Tuple

class SVMI:
    def __init__(self, C=1, kernel='rbf', gamma='auto', random_state=None):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state


    def _generate_V_matrix(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # max_dims are the upper bounds of each dimension.
        # Since there is not clear specification on which values have to be picked,
        # using the maximum for each dimension should be a good choice
        max_dims = np.max(X, axis=0)

        V = np.array(
            [[np.prod(max_dims - np.maximum(x_i, x_j)) for x_j in X] 
            for x_i in X]
        )

        return V
    

    def _simple_inference(
        self,
        K: npt.NDArray[np.float64],
        use_v_matrix=False,
        verbose=False
    ) -> Tuple[npt.NDArray[np.float64], float]:
        # Compute V matrix and Gramm matrix
        V = self._generate_V_matrix(self.X) if use_v_matrix else np.eye(self.l)

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

        if verbose:
            print(f'c: {c}')
            print(f'A: {A}')

        return A, c
    

    def _invariance_inference(
        self,
        K: npt.NDArray[np.float64],
        invariant_funcs: List[Callable],
        use_v_matrix=False,
        verbose=False
    ) -> Tuple[npt.NDArray[np.float64], float]:
        # Compute V matrix and Gramm matrix
        V = self._generate_V_matrix(self.X) if use_v_matrix else np.eye(self.l)

        # Create auxiliar variables
        ones = np.ones(self.l)
        VK = np.dot(V, K)
        VK_perturbed_inv = np.linalg.inv(VK + self.C * np.eye(self.l))

        # Compute invariants and store them as columns in a 2D array
        invariant_args = {'X': self.X, 'y': self.y}
        invariants = np.array([func(**invariant_args) for func in invariant_funcs])

        # Compute vectors
        # A_s is a 2D array whose columns contain the individual A_s_i values
        A_v = np.dot(VK_perturbed_inv, np.dot(V, y))
        A_c = np.dot(VK_perturbed_inv, np.dot(V, ones))
        A_s = np.array([np.dot(VK_perturbed_inv, phi) for phi in invariants])

        # Create system of equations
        c_1 = np.dot(ones, np.dot(VK, A_c)) - np.dot(ones, np.dot(V, ones))
        mu_1 = np.array([np.dot(ones, np.dot(VK, phi)) - np.dot(ones, phi) for phi in invariants])
        rh_1 = np.dot(ones, np.dot(VK, A_v)) - np.dot(ones, np.dot(V, self.y))

        c_2 = np.array([np.dot(A_c, np.dot(K, phi)) - np.dot(ones, phi) for phi in invariants])
        mu_2 = np.array([np.array([np.dot(A_s[s], np.dot(K, invariants[k])) for s in range(len(invariant_funcs))]) for k in range(len(invariant_funcs))])
        rh_2 = np.array([np.dot(A_v, np.dot(K, phi)) - np.dot(self.y, phi) for phi in invariants])


        a_1 = np.concatenate(([c_1], mu_1))
        a_2 = np.vstack(([c_2], mu_2.T)).T
        a = np.concatenate(([a_1], a_2), axis=0)
        b = np.concatenate(([rh_1], rh_2))

        solution = np.linalg.solve(a, b)

        c, mu = solution[0], solution[1:]

        # The sum can be replaced with a dot product
        A = A_v - c * A_c - np.sum(np.array([mu[s] * A_s[s] for s in range(len(invariant_funcs))]), axis=0)

        if verbose:
            print('Invariants weights: ', mu)
            print(f'c: {c}')
            print(f'A: {A}')

        return A, c


    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        use_v_matrix=False,
        invariant_funcs=None,
        verbose=False
    ):
        self.X = X
        self.y = y
        self.l = len(y)
        self.d = X.shape[1]

        np.random.seed(self.random_state)

        if self.gamma == 'auto':
            self.gamma = 1 / self.d
        
        if self.kernel == 'rbf':
            self.kernel = rbf_kernel
            K = self.kernel(X, X, gamma=self.gamma)
        elif self.kernel == 'linear':
            self.kernel = linear_kernel
            K = self.kernel(X, X)

        if invariant_funcs is None:
            A, c = self._simple_inference(K, use_v_matrix=use_v_matrix, verbose=verbose)
        else:
            A, c = self._invariance_inference(K, invariant_funcs, use_v_matrix=use_v_matrix, verbose=verbose)
        
        self.A, self.c = A, c


    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        # Apply decision rule
        prediction = np.dot(self.A, rbf_kernel(self.X, X, gamma=self.gamma)) + self.c

        # Get label: if the result of the decision rule is smaller than 0.5, then
        # it's classified as 0. Otherwise, it's classified as 1
        prediction = np.where(prediction < 0.5, 0, 1).flatten()

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

