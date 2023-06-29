from cgi import test
import sys
sys.path.append("../")

import numpy as np
from scipy.optimize import minimize
from typing import Any
from typing import Iterable, Union, cast, Tuple

from quri_parts.core.operator import Operator
from quri_parts.core.estimator.sampling.estimator import _Estimate
from quri_parts.core.state import GeneralCircuitQuantumState, ParametricCircuitQuantumState
from quri_parts.core.estimator.sampling.pauli import (
    general_pauli_sum_expectation_estimator,
    general_pauli_sum_sample_variance,
    general_pauli_covariance_estimator,
)
from quri_parts.core.sampling.shots_allocator import (
    create_proportional_shots_allocator,
    create_equipartition_shots_allocator
)

from utils.challenge_2023 import QuantumCircuitTimeExceededError
from common import (
    prepare_problem, prepare_ansatz,
    prepare_zne_estimator, prepare_noiseless_estimator,
    CostEvaluator,
    challenge_sampling,
    noiseless_sampling
)
from FourierAnsatz import FourierAnsatz, param_convert_func_FourierAnsatz
from CosineSum.CosineSumGenerator import GenerateCosineSumInstance
from CosineSum.CosineSumSolver_CG import CosineSumSolver_CG
from measurement import Measure

np.set_printoptions(formatter={'float': '{: 0.8f}'.format}, linewidth=10000)

"""
####################################
add codes here
####################################
"""

class Evaluator(Measure):
    def __init__(
        self, noise: bool, hardware_type: str,
        hamiltonian: Operator, ansatz: FourierAnsatz,
        total_shots: int,
        total_measures: int = 1,
        optimization_level: int = 0,
        bit_flip_error: float = 0
    ):
        super().__init__(
            noise, hardware_type,
            hamiltonian, ansatz,
            total_shots,
            optimization_level, bit_flip_error
        )
        self.total_measures = total_measures
        self.params_list: list[list[float]] = []
        self.cost_list: list[float] = []
        self.shot_error_list: list[float] = []
        self.error_list: list[float] = []
    
    def reset(self):
        self.params_list = []
        self.cost_list = []
        self.shot_error_list = []
        self.error_list = []

    def measure(self, params: list[float], show_output: bool = False) -> Tuple[float, float]:
        challenge_sampling.reset()
        noiseless_sampling.reset()
        
        costs, shot_errors = [], []
        for _ in range(self.total_measures):
            cost, shot_error = super().measure(params)
            costs.append(cost)
            shot_errors.append(shot_error)
        cost = np.array(costs).mean()
        shot_error = np.array(shot_errors).mean()
        error = np.array(costs).std()

        if(show_output and len(self.params_list) == 0):
            time = max(challenge_sampling.total_quantum_circuit_time, noiseless_sampling.total_quantum_circuit_time)
            print(self.hardware_type, self.total_shots, self.total_measures, ": time =", time)

        self.params_list.append(params)
        self.cost_list.append(cost)
        self.shot_error_list.append(shot_error)
        self.error_list.append(error)
        return cost, error
    
    # cost + error * error_coef が最小となるindexを返す
    def min_idx(self, error_coef: float = 0) -> int:
        min_idx = 0
        for i in range(len(self.cost_list)):
            if(self.cost_list[i] + self.error_list[i] * error_coef < self.cost_list[min_idx] + self.error_list[min_idx] * error_coef):
                min_idx = i
        return min_idx
    
    # cost + error * error_coef の最小値を返す
    def min_value(self, error_coef: float = 0) -> float:
        min_idx = self.min_idx(error_coef)
        return self.cost_list[min_idx] + self.error_list[min_idx] * error_coef
    
    # cost + error * error_coef を最小にするパラメータを返す
    def min_value_params(self, error_coef: float = 0) -> list[float]:
        min_idx = self.min_idx(error_coef)
        return self.params_list[min_idx]

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
        ansatz_sc, ansatz_it = prepare_ansatz("sc", n_qubits), prepare_ansatz("it", n_qubits)

        evaluator_precise = Evaluator(
            False, "it",
            hamiltonian, ansatz_it,
            100000, 1
        )
        evaluator_casual = Evaluator(
            True, "it",
            hamiltonian, ansatz_it,
            100, 10
        )

        evaluators: list[Evaluator] = [
            # Evaluator(True, "sc", hamiltonian, ansatz_sc, 10000, 10), # 2.1s (3 layers),  2.6s (4 layers)
            # Evaluator(True, "sc", hamiltonian, ansatz_sc, 5000, 2), # 0.21s (3 layers),  0.26s (4 layers)
            # Evaluator(True, "sc", hamiltonian, ansatz_sc, 3333, 3),
            # Evaluator(True, "sc", hamiltonian, ansatz_sc, 2000, 5),
            # Evaluator(True, "sc", hamiltonian, ansatz_sc, 1000, 10),
            # Evaluator(True, "it", hamiltonian, ansatz_it, 100, 10), # 0.252s (3 layers), 0.318s (4 layers)
            # Evaluator(True, "it", hamiltonian, ansatz_it, 300, 10), # 2.422s (3 layers), 3.010s (4 layers)
            # Evaluator(True, "it", hamiltonian, ansatz_it, 500, 10), # 4.952s (3 layers), 6.146s (4 layers)
            # Evaluator(True, "it", hamiltonian, ansatz_it, 1000, 10), # 11.218s (3 layers), 13.912s (4 layers)
        ]

        # 0.252 * 20 * 12 * 1 * 4 = 242.88s (3 layers)
        # 11.218 * 20 * 12 = 2692.32s (3 layers)

        num_init_points = 1
        num_loop_counts = 3
        num_params = ansatz_sc.parameter_count
        for init_point_idx in range(num_init_points):
            print("new generation")
            evaluator_casual.reset()
            evaluator_precise.reset()
            current_params = (np.random.rand(num_params) * 2 * np.pi).tolist()
            for loop_idx in range(num_loop_counts):
                for param_idx in range(0, num_params, 3): # 12 loops (3 layers), 16 loops (4 layers)
                    cosineSumInstance = GenerateCosineSumInstance(
                        3, True,
                        lambda params: evaluator_precise.measure(
                            current_params[: param_idx] + params + current_params[param_idx + 3 :]
                        )[0],
                        # lambda params: evaluator_casual.measure(
                        #     current_params[: param_idx] + params + current_params[param_idx + 3 :]
                        # )[0],
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
                    # print("cosine_sum :", cosineSumInstance.eval(optimal_params))

                    # test_evaluator_params = evaluator_precise.params_list[-1]
                    # test_cosineSum_params = test_evaluator_params[param_idx : param_idx + 3]
                    # print(test_cosineSum_params)
                    # print("cosineSum", cosineSumInstance.eval(test_cosineSum_params))
                    # print(test_evaluator_params)
                    # print("precise", evaluator_precise.measure(test_evaluator_params))
                    # exit()

                    casual_cost, _ = evaluator_casual.measure(current_params)
                    print("casual :", casual_cost)
                    precise_cost, _ = evaluator_precise.measure(current_params)
                    print("precise :", precise_cost)
                    # for evaluator in evaluators:
                    #     cost, error = evaluator.measure(current_params)
                    #     error = error / np.sqrt(evaluator.total_measures - 1)
                    #     diff = cost - precise_cost
                    #     score = -diff**2 / (2 * error**2) - np.log(error)
                    #     print(evaluator.hardware_type, evaluator.total_shots, evaluator.total_measures, ":", f"({score})", cost, "+-", error)
                    #     print(" elapsed time =", challenge_sampling.total_quantum_circuit_time)
                    # print(current_params)
                    print()
            # print("min cost =", evaluator_casual.min_value())
            # print("param =", evaluator_casual.min_value_params())
            print("min cost =", evaluator_precise.min_value())
            print("param =", evaluator_precise.min_value_params())
            print()
        # print(evaluator_precise.min_value())
        # print(challenge_sampling.total_quantum_circuit_time)
        return evaluator_precise.min_value()

        # optimize
        n_init_params = 3
        ans_cost, ans_params = 0, np.zeros(ansatz._circuit.parameter_count)
        for itr in range(n_init_params):
            print("iteration:", itr)
            remaining_time = 1000 - challenge_sampling.total_quantum_circuit_time
            allocated_time = remaining_time / (n_init_params - itr)
            time_end_approx = challenge_sampling.total_quantum_circuit_time + allocated_time * 0.3
            time_end_precise = challenge_sampling.total_quantum_circuit_time + allocated_time

            init_params = np.random.rand(ansatz._circuit.parameter_count) * 2 * np.pi
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
    for i in range(10):
        print(run_algorithm.get_result())
