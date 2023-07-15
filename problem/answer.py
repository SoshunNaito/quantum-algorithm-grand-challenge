import sys
sys.path.append("../")
import numpy as np
from typing import Any
from typing import Union, Tuple

from quri_parts.core.operator import Operator

from common import (
    prepare_problem,
    challenge_sampling
)
from GivensAnsatz import (
    GivensAnsatz,
    GivensAnsatz_it_8,
    GivensAnsatzOptimizer
)
from measurement import Measure

np.set_printoptions(formatter={'float': '{: 0.8f}'.format}, linewidth=10000)

"""
####################################
add codes here
####################################
"""

class Evaluator(Measure):
    def __init__(
        self, hardware_type: str,
        hamiltonian: Operator, ansatz: GivensAnsatz,
        total_shots: int,
        total_measures: int = 1,
        optimization_level: int = 0,
        bit_flip_error: float = 0
    ):
        super().__init__(
            hardware_type,
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

    def measure(self, params: list[float]) -> Tuple[float, float]:
        costs, shot_errors = [], []
        for _ in range(self.total_measures):
            cost, shot_error = super().measure(params)
            costs.append(cost)
            shot_errors.append(shot_error)
        cost = np.array(costs).mean()
        shot_error = np.array(shot_errors).mean()
        error = np.array(costs).std()

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

        def measurement_func(evaluator: Evaluator, params: list[float]) -> Tuple[float, float]:
            cost, error = evaluator.measure(params)
            print("", cost, "+-", error)
            return cost, error
        
        freeze_evaluation_results: dict[Tuple[Union[Tuple[int, int], None], Union[Tuple[int, int], None]], Evaluator] = {}
        for freeze_ones in [None, (0, 1), (2, 3), (4, 5), (6, 7)]:
            for freeze_zeros in [None, (0, 1), (2, 3), (4, 5), (6, 7)]:
                if(freeze_ones == freeze_zeros): continue
                print("ones =", freeze_ones, "  zeros =", freeze_zeros)

                list_ones = list(freeze_ones) if freeze_ones is not None else []
                list_zeros = list(freeze_zeros) if freeze_zeros is not None else []
                ansatz = GivensAnsatz_it_8(list_ones, list_zeros)
                evaluator = Evaluator( # 0.531 sec
                    "it", hamiltonian, ansatz,
                    total_shots = 100,
                    total_measures = 10,
                    bit_flip_error = 1e-3
                )
                for _ in range(10):
                    params = (np.random.rand(ansatz.num_parameters) * 2 * np.pi).tolist()
                    cost, error = measurement_func(evaluator, params)
                freeze_evaluation_results[(freeze_ones, freeze_zeros)] = evaluator
        
        min_energy_results = {}
        for configure, evaluator in freeze_evaluation_results.items():
            min_energy_results[configure] = evaluator.min_value()
        print(min_energy_results)

        print("min cost :", min(min_energy_results.values()))
        print("elapsed time =", challenge_sampling.total_quantum_circuit_time)
        print()

        # get sorted result
        K = 5
        top_K = sorted(
            freeze_evaluation_results.items(),
            key=lambda item: np.array(item[1].cost_list).mean()
        )[:K]
        for configure, evaluator in top_K:
            freeze_ones, freeze_zeros = configure
            print("ones =", freeze_ones, "  zeros =", freeze_zeros)

            list_ones = list(freeze_ones) if freeze_ones is not None else []
            list_zeros = list(freeze_zeros) if freeze_zeros is not None else []
            ansatz = GivensAnsatz_it_8(list_ones, list_zeros)
            optimizer = GivensAnsatzOptimizer(ansatz)
            optimizer.optimize(
                measurement_func = lambda params: measurement_func(evaluator, params)[0],
                init_params = evaluator.min_value_params(),
                num_iterations = 2,
                num_additional_measure_1 = 5,
                num_additional_measure_3 = 17
            )
            print("min cost", freeze_ones, freeze_zeros, ":", evaluator.min_value())
            print("elapsed time =", challenge_sampling.total_quantum_circuit_time)
            print()
        best_configure, best_evaluator = top_K[np.argmin([evaluator.min_value() for _, evaluator in top_K])]
        ans = best_evaluator.min_value()
        print("ans =", ans)
        return ans

if __name__ == "__main__":
    run_algorithm = RunAlgorithm()
    print(run_algorithm.get_result())