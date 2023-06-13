import numpy as np
from typing import Tuple
from numpy.typing import NDArray

from CosineSumInstance import CosineSumInstance
from CosineSumGenerator import GenerateCosineSumInstance
from CosineSumSolver_CG import CosineSumSolver_CG

def eval(
    N: int, thetas: list[float],
    quad_amplitudes: NDArray, quad_offsets: NDArray,
    linear_amplitudes: NDArray, linear_offsets: NDArray,
) -> float:
    ans = 0
    for i in range(N):
        for j in range(i + 1, N):
            ans += quad_amplitudes[i, j] * np.cos(thetas[i] - thetas[j] + quad_offsets[i, j])
        ans += linear_amplitudes[i] * np.cos(thetas[i] + linear_offsets[i])
    return ans

def grad(
    N: int, thetas: list[float],
    quad_amplitudes: NDArray, quad_offsets: NDArray,
    linear_amplitudes: NDArray, linear_offsets: NDArray,
) -> NDArray:
    ans = np.zeros(N)
    for i in range(N):
        for j in range(i + 1, N):
            tmp = quad_amplitudes[i, j] * np.sin(thetas[i] - thetas[j] + quad_offsets[i, j])
            ans[i] -= tmp
            ans[j] += tmp
        ans[i] -= linear_amplitudes[i] * np.sin(thetas[i] + linear_offsets[i])
    return ans

def hess(
    N: int, thetas: list[float],
    quad_amplitudes: NDArray, quad_offsets: NDArray,
    linear_amplitudes: NDArray, linear_offsets: NDArray,
) -> NDArray:
    ans = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            tmp = quad_amplitudes[i, j] * np.cos(thetas[i] - thetas[j] + quad_offsets[i, j])
            ans[i, j] = tmp
            ans[j, i] = tmp
            ans[i, i] -= tmp
            ans[j, j] -= tmp
        ans[i, i] -= linear_amplitudes[i] * np.cos(thetas[i] + linear_offsets[i])
    return ans

class Result:
    def __init__(self, thetas: NDArray, energy: float, _hess_min_eigval: float):
        self._thetas: NDArray = thetas
        self._energy: float = energy
        self._hess_min_eigval: float = 0.0
        self._count: int = 1


n_samples: int = 100
n_points: int = 20
local_minima_counts: dict[Tuple[int, int], int] = dict()

for N in [2, 4, 8, 16]: # num of thetas (= num of variables + 1)
    print("N =", N)

    ans_results: list[Result] = []
    ans_amplitudes: NDArray = np.zeros((N, N))
    ans_offsets: NDArray = np.zeros((N, N))

    counts: dict[int, int] = dict()
    for sample_idx in range(n_samples):
        quad_amplitudes: NDArray = np.random.sample(size = N ** 2).reshape((N, N)) * 2 - 1
        quad_offsets: NDArray = np.random.sample(size = N ** 2).reshape((N, N)) * (np.pi * 2)
        linear_amplitudes: NDArray = np.random.sample(size = N) * 2 - 1
        linear_offsets: NDArray = np.random.sample(size = N) * (np.pi * 2)

        for i in range(N):
            quad_amplitudes[i, i] = 0
            quad_offsets[i, i] = 0
            for j in range(i + 1, N):
                quad_amplitudes[i, j] = quad_amplitudes[j, i]
                quad_offsets[i, j] = -quad_offsets[j, i]
        
        instance = GenerateCosineSumInstance(
            N, True,
            lambda params: eval(N, params, quad_amplitudes, quad_offsets, linear_amplitudes, linear_offsets),
            3
        )

        for itr in range(n_points):
            theta_params = np.random.sample(N) * (np.pi * 2)
            eval_ref = eval(N, theta_params.tolist(), quad_amplitudes, quad_offsets, linear_amplitudes, linear_offsets)
            eval_test = instance.eval(theta_params.tolist())
            if(abs(eval_ref - eval_test) > 1e-8):
                print("ERROR at eval")
                print(eval_ref, eval_test)
                exit()
            
            grad_ref = grad(N, theta_params.tolist(), quad_amplitudes, quad_offsets, linear_amplitudes, linear_offsets)
            grad_test = np.array(instance.grad(theta_params.tolist()))
            if(np.linalg.norm(grad_ref - grad_test) > 1e-8):
                print("ERROR at grad")
                print(grad_ref, grad_test)
                exit()
            
            hess_ref = hess(N, theta_params.tolist(), quad_amplitudes, quad_offsets, linear_amplitudes, linear_offsets)
            hess_test = np.array(instance.hess(theta_params.tolist()))
            if(np.linalg.norm(hess_ref - hess_test) > 1e-8):
                print("ERROR at hess")
                print(hess_ref, hess_test)
                exit()