import sys
sys.path.append("../")

import numpy as np
from scipy.optimize import minimize
from typing import Any

from utils.challenge_2023 import QuantumCircuitTimeExceededError
from common import (
    prepare_problem, prepare_ansatz, prepare_sampling_estimator,
    CostEvaluator,
    challenge_sampling,
)
from FourierAnsatz import param_convert_func_FourierAnsatz
from CosineSum.CosineSumGenerator import GenerateCosineSumInstance
from CosineSum.CosineSumSolver_CG import CosineSumSolver_CG

np.set_printoptions(formatter={'float': '{: 0.8f}'.format}, linewidth=10000)

"""
####################################
add codes here
####################################
"""

class RunAlgorithm:
    def __init__(self) -> None:
        challenge_sampling.reset()

    def result_for_evaluation(self) -> tuple[Any, float]:
        energy_final = self.get_result()
        qc_time_final = challenge_sampling.total_quantum_circuit_time

        return energy_final, qc_time_final

    def get_result(self) -> float:
        """
        ####################################
        add codes here
        ####################################
        """

        n_qubits = 8
        hamiltonian = prepare_problem(n_qubits)
        parametric_state_sc, parametric_state_it = prepare_ansatz("sc", n_qubits), prepare_ansatz("it", n_qubits)
        sampling_estimator_sc_1000 = prepare_sampling_estimator("sc", 1000)
        sampling_estimator_it_1000 = prepare_sampling_estimator("it", 500)

        evaluator_casual = CostEvaluator(hamiltonian, parametric_state_sc, sampling_estimator_sc_1000, param_convert_func_FourierAnsatz)
        evaluator_formal = CostEvaluator(hamiltonian, parametric_state_it, sampling_estimator_it_1000, param_convert_func_FourierAnsatz)

        num_init_points = 1
        num_loop_counts = 13
        num_params = parametric_state_sc._circuit.parameter_count
        for init_point_idx in range(num_init_points):
            current_params = (np.random.rand(num_params) * 2 * np.pi).tolist()
            for loop_idx in range(num_loop_counts):
                for param_idx in range(0, num_params, 3):
                    evaluator = evaluator_formal
                    cosineSumInstance = GenerateCosineSumInstance(
                        3, True,
                        lambda params: evaluator.eval(
                            current_params[: param_idx] + params + current_params[param_idx + 3 :],
                            # show_output=True
                        ),
                        7 # 13 + 7 = 20 evaluations
                    )
                    # 4 * (10 * 0.08534399999999999 + 1 * 1.1637) * 6 * 20 = 1000
                    optimal_params = CosineSumSolver_CG(cosineSumInstance).solve(
                        [
                            (np.random.rand(3) * 2 * np.pi).tolist()
                            for _ in range(100)
                        ], True
                    )
                    current_params[param_idx : param_idx + 3] = optimal_params
                    print(cosineSumInstance.eval(optimal_params), "vs", evaluator_casual.eval(current_params), "vs", evaluator_formal.eval(current_params))
            print()
        print(min(evaluator_casual.min_cost, evaluator_formal.min_cost))
        print(challenge_sampling.total_quantum_circuit_time)
        exit()

        # optimize
        n_init_params = 3
        ans_cost, ans_params = 0, np.zeros(parametric_state._circuit.parameter_count)
        for itr in range(n_init_params):
            print("iteration:", itr)
            remaining_time = 1000 - challenge_sampling.total_quantum_circuit_time
            allocated_time = remaining_time / (n_init_params - itr)
            time_end_approx = challenge_sampling.total_quantum_circuit_time + allocated_time * 0.3
            time_end_precise = challenge_sampling.total_quantum_circuit_time + allocated_time

            init_params = np.random.rand(parametric_state._circuit.parameter_count) * 2 * np.pi
            print("<< approx >>")
            try:
                evaluator_approx.reset()
                result = minimize(
                    lambda params: evaluator_approx.eval(params, time_end_approx, False, True),
                    init_params,
                    method = "Powell",
                )
            except QuantumCircuitTimeExceededError: pass
            ans_cost, ans_params = evaluator_approx.min_cost, evaluator_approx.min_params
            
            print("<< precise >>")
            try:
                evaluator_precise.reset()
                result = minimize(
                    lambda params: evaluator_precise.eval(params, time_end_precise, False, True),
                    ans_params,
                    method = "Powell",
                )
            except QuantumCircuitTimeExceededError: pass
            ans_cost, ans_params = evaluator_precise.min_cost, evaluator_precise.min_params
            print()
        return ans_cost


if __name__ == "__main__":
    run_algorithm = RunAlgorithm()
    print(run_algorithm.get_result())
