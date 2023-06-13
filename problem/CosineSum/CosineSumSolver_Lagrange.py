import numpy as np
import itertools
import math

from scipy.optimize import minimize
from .CosineSumInstance import CosineSumInstance


class CosineSumSolver_Lagrange:
    def __init__(self, instance: CosineSumInstance):
        self.instance = instance

    def solve(self, init_lambdas: list[float]) -> list[float]:
        assert(np.allclose(self.instance.coefficient_matrix, self.instance.coefficient_matrix.T))
        N = self.instance.num_theta_params

        converged = False
        lambdas = np.array(init_lambdas, dtype = np.float64)
        while(not converged):
            Lagrange_matrix = self.instance.coefficient_matrix + np.diag(
                list(itertools.chain.from_iterable([(l, l) for l in lambdas]))
            )
            L, V = np.linalg.eigh(Lagrange_matrix)
            V = [V[:, i] for i in range(len(L))]
            
            Derivative_matrix = np.zeros((N, N))
            for j in range(1, len(L)):
                P = np.array([V[0][k * 2] * V[j][k * 2] + V[0][k * 2 + 1] * V[j][k * 2 + 1] for k in range(N)])
                Derivative_matrix += np.outer(P, P) * 2 / (L[0] - L[j])
            diff = np.array([
                1 / N - V[0][k * 2] ** 2 - V[0][k * 2 + 1] ** 2 if (V[0][k * 2] ** 2 + V[0][k * 2 + 1] ** 2) * N > 0.1
                else 0 for k in range(N)
            ])

            if(np.linalg.norm(diff) < 0.001):
                converged = True
                break

            delta_lambdas = np.linalg.pinv(Derivative_matrix) @ diff * 0.1
            delta_lambdas -= delta_lambdas.mean()
            lambdas += delta_lambdas

            print(' '.join(['{:.4f}'.format(l) for l in L]))
            print(' ' + ' '.join(['{:.4f}'.format(v) for v in V[0]]))
            print('  ' + ' '.join(['{:.4f}'.format(d) for d in diff]))

            # exit(0)
        
        ans = [math.atan2(V[0][i * 2 + 1], V[0][i * 2]) for i in range(N)]
        return ans