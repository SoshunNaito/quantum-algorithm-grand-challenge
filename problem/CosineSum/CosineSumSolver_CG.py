import sys
sys.path.append("../")

import numpy as np
from scipy.optimize import minimize
from nptyping import NDArray

from CosineSum.CosineSumInstance import CosineSumInstance

class CosineSumSolver_CG:
    def __init__(self, instance: CosineSumInstance):
        self.instance = instance

    def solve(self, init_theta_params_list: list[list[float]], has_linear_terms: bool) -> list[float]:
        for init_theta_params in init_theta_params_list:
            assert len(init_theta_params) == self.instance.num_theta_params

        thresh = 0.01
        # np.allclose(atol=thresh)
        results: list[NDArray] = []
        counts: list[int] = []

        energy, ans = None, [0] * self.instance.num_theta_params
        for init_theta_params in init_theta_params_list:
            res = minimize(
                self.instance.eval,
                init_theta_params,
                method = "Newton-CG",
                jac = self.instance.grad,
                hess = self.instance.hess,
            )
            tmp = np.array(res.x)
            if(has_linear_terms == False):
                tmp -= tmp[0]

            _energy = self.instance.eval(tmp.tolist())
            if energy == None or _energy < energy:
                energy, ans = _energy, tmp.tolist()

            found = False
            for i, result in enumerate(results):
                diff = (result - tmp) / (2 * np.pi)
                diff -= np.round(diff)

                if np.allclose(diff, np.zeros(self.instance.num_theta_params), atol=thresh):
                    counts[i] += 1
                    found = True
                    break
            if not found:
                results.append(tmp)
                counts.append(1)
        
        # for i in range(len(results)):
        #     print(f"{results[i] / (2 * np.pi)} ({self.instance.eval(list(results[i]))}): {counts[i]}")
        # print("".join(["  "] * len(results)) + str(len(results)))
        # print()
        for i in range(len(ans)):
            ans[i] -= np.round(ans[i] / (np.pi * 2)) * (np.pi * 2)
        return ans