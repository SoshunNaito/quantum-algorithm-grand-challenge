from math import e
import numpy as np
from quri_parts.core.operator import Operator
from quri_parts.core.estimator.sampling.estimator import _Estimate
from quri_parts.core.state import GeneralCircuitQuantumState, ParametricCircuitQuantumState
from quri_parts.core.estimator.sampling.pauli import (
    general_pauli_sum_expectation_estimator,
    general_pauli_sum_sample_variance,
    general_pauli_covariance_estimator,
)
from collections import defaultdict
from collections.abc import Mapping
from typing import Union, cast, Tuple

import math
from collections.abc import Collection

from numpy.random import default_rng

from quri_parts.core.operator import CommutablePauliSet, Operator
from quri_parts.core.sampling import PauliSamplingSetting, PauliSamplingShotsAllocator

import numpy as np

from quri_parts.core.measurement import (
    PauliReconstructorFactory,
    bitwise_pauli_reconstructor_factory,
)
from quri_parts.core.operator import PAULI_IDENTITY, CommutablePauliSet, PauliLabel
from quri_parts.core.sampling import MeasurementCounts
from quri_parts.core.utils.statistics import count_var
from common import (
    prepare_problem, prepare_ansatz,
    prepare_sampling_estimator, prepare_zne_estimator, prepare_noiseless_estimator,
    CostEvaluator,
    challenge_sampling,
    noiseless_sampling
)
from FourierAnsatz import param_convert_func_FourierAnsatz
from measurement import Measure

def estimate_variance_from_measurement(
    counts: MeasurementCounts,
    pauli_set: CommutablePauliSet,
    coefs: Mapping[PauliLabel, complex],
    reconstructor_factory: PauliReconstructorFactory,
) -> float:
    ans = 0
    for pauli_1 in pauli_set:
        coef1 = coefs[pauli_1].real
        for pauli_2 in pauli_set:
            coef2 = coefs[pauli_2].real
            cov = general_pauli_covariance_estimator(counts, pauli_1, pauli_2, reconstructor_factory)
            ans += coef1 * coef2 * cov
    return ans

def get_measurement_info(estimate: _Estimate) -> Tuple[dict[CommutablePauliSet, int], dict[CommutablePauliSet, float]]:
    count_dict, var_dict = {}, {}
    for pauli_set, pauli_rec, counts in zip(
        estimate._pauli_sets, estimate._pauli_recs, estimate._sampling_counts
    ):
        # # counts: 偶数個のビットが立っていたら同符号、奇数個のビットが立っていたら異符号
        # coefs = [estimate._op[pauli].real for pauli in pauli_set]
        # mean = general_pauli_sum_expectation_estimator(
        #     counts, pauli_set, estimate._op, pauli_rec
        # ).real
        # var = general_pauli_sum_sample_variance(
        #     counts, pauli_set, estimate._op, pauli_rec
        # ) / sum(counts.values())
        count_dict[pauli_set] = sum(counts.values())
        est_var = estimate_variance_from_measurement(counts, pauli_set, estimate._op, pauli_rec)
        var_dict[pauli_set] = est_var
    return count_dict, var_dict

def create_variance_proportional_shots_allocator(
    count_dict: dict[CommutablePauliSet, int], var_dict: dict[CommutablePauliSet, float]
) -> PauliSamplingShotsAllocator:

    def allocator(
        operator: Operator, pauli_sets: Collection[CommutablePauliSet], total_shots: int
    ) -> Collection[PauliSamplingSetting]:
        pauli_sets = tuple(pauli_sets)  # to fix the order of elements

        var_sum = sum(var_dict[pauli_set] for pauli_set in pauli_sets)

        ratios = [var_dict[pauli_set] / var_sum for pauli_set in pauli_sets]
        shots_list = [np.floor(total_shots * ratio) for ratio in ratios]
        return frozenset(
            {
                PauliSamplingSetting(
                    pauli_set=pauli_set,
                    n_shots=n_shots,
                )
                for (pauli_set, n_shots) in zip(pauli_sets, shots_list)
            }
        )

    return allocator

class MeasureBatchExecutor:
    def __init__(self, measures: dict[str, Measure], num_measurement: int) -> None:
        self.measures = measures
        self.num_measurement = num_measurement

    def execute(self, params: list[float]):
        for name, measure in self.measures.items():
            print(name)
            costs, shot_errors = [], []
            for i in range(self.num_measurement):
                challenge_sampling.reset()
                noiseless_sampling.reset()
                cost, shot_error = measure.measure(param_convert_func_FourierAnsatz(params))
                costs.append(cost)
                shot_errors.append(shot_error)

                if(i == 0):
                    circuit_time = max(challenge_sampling.total_quantum_circuit_time, noiseless_sampling.total_quantum_circuit_time)
                    print(" time:", circuit_time)
                print(f" {i}:", cost, "+-", shot_error)
            print(" cost:", np.array(costs).mean(), "+-", np.array(costs).std(), f"shot_error = {np.array(shot_errors).mean()}")

class SamplingTester:
    def __init__(
        self, hamiltonian: Operator,
        parametric_state_sc: ParametricCircuitQuantumState,
        parametric_state_it: ParametricCircuitQuantumState,
        num_shots_sc: list[int],
        num_shots_it: list[int],
        num_measurement: int
    ):
        self.hamiltonian = hamiltonian
        # self.parametric_states = {
        #     "sc": parametric_state_sc,
        #     "it": parametric_state_it,
        # }
        # self.estimators = {}
        # self.evaluators = []
        self.measures: dict[str, Measure] = {}

        # self.estimators["noiseless"] = prepare_noiseless_estimator("it", 10000)
        # noiseless_evaluator = CostEvaluator(
        #     "noiseless", "it",
        #     lambda name, hardware_type, params: self.estimators[name](
        #         self.hamiltonian, parametric_state_it, [param_convert_func_FourierAnsatz(params)]
        #     )[0]
        # )
        # self.evaluators.append(noiseless_evaluator)

        
        self.measures[f"<< noiseless >>"] = Measure(
            False, "it", 750000,
            self.hamiltonian, parametric_state_it,
        )

        for (hardware_type, num_shots_list) in [("sc", num_shots_sc), ("it", num_shots_it)]:
            parametric_state = parametric_state_sc if hardware_type == "sc" else parametric_state_it
            for num_shots in num_shots_list:
                # self.estimators[f"{hardware_type} {num_shots}"] = prepare_sampling_estimator(hardware_type, num_shots)
                # sampling_evaluator = CostEvaluator(
                #     f"{hardware_type} {num_shots}", hardware_type,
                #     lambda name, hardware_type, params: self.estimators[name](
                #         self.hamiltonian, self.parametric_states[hardware_type], [param_convert_func_FourierAnsatz(params)]
                #     )[0]
                # )
                # self.evaluators.append(sampling_evaluator)

                for noise in [False, True]:
                    noise_str = "noise" if noise else ""
                    for mitigate in [False, True]:
                        if(noise == False and mitigate == True):
                            continue
                        mitigate_str = "mitigate" if mitigate else ""
                        self.measures[f"{hardware_type} {num_shots} {noise_str} {mitigate_str}"] = Measure(
                            noise, hardware_type, num_shots,
                            self.hamiltonian, parametric_state,
                            0,
                            (1e-2 if hardware_type == "sc" else 1e-3) if (noise and mitigate) else 0
                        )

                # self.estimators[f"{hardware_type} {num_shots} zne"] = prepare_zne_estimator(hardware_type, num_shots)
                # zne_evaluator = CostEvaluator(
                #     f"{hardware_type} {num_shots} zne", hardware_type,
                #     lambda name, hardware_type, params: self.estimators[name](
                #         self.hamiltonian, self.parametric_states[hardware_type].bind_parameters(param_convert_func_FourierAnsatz(params))
                #     )
                # )
                # self.evaluators.append(zne_evaluator)
        self.batchExecutor = MeasureBatchExecutor(self.measures, num_measurement)
    
    def test(self, params: list[float]):
        self.batchExecutor.execute(params)

# HardwareEfficientReal(qubit_count=4, reps=1)
# qubit count: 4
# circuit depth: 5
# parameter count: 8
# sc 100 : 0.41132356654598434
# sc 300 : 0.31696992005260344 # この辺までかなぁ
# sc 1000 : 0.26288519755878254 # この辺は厳しそう
# sc 3000 : 0.29550985171187283
# it 100 : 0.27372005338326355
# it 300 : 0.16175356874089117
# it 1000 : 0.09226362853561562
# it 3000 : 0.057907271784574485
# it 10000 : 0.03538879474116862 # 全然いける

# HardwareEfficientReal(qubit_count=4, reps=2)
# qubit count: 8
# circuit depth: 8
# parameter count: 12
# sc 100 : 0.4434873620324149 # この辺までかなぁ
# sc 300 : 0.3824302617420672 # この辺は厳しそう
# sc 1000 : 0.35186661022848487
# sc 3000 : 0.3797425070941567
# it 100 : 0.3028266469471391
# it 300 : 0.16344085218195897
# it 1000 : 0.09486215285589274
# it 3000 : 0.058184030270776
# it 10000 : 0.03609122147036358

# HardwareEfficientReal(qubit_count=4, reps=3)
# qubit count: 4
# circuit depth: 11
# parameter count: 16
# it 100 : 0.3065444331167448
# it 300 : 0.17223464391229593
# it 1000 : 0.1076536365078404
# it 3000 : 0.06202352389700072
# it 10000 : 0.042076192686706065

# HardwareEfficientReal(qubit_count=4, reps=4)
# qubit count: 4
# circuit depth: 14
# parameter count: 20
# it 100 : 0.298585799231483
# it 300 : 0.16667379695355108
# it 1000 : 0.09405239625872107
# it 3000 : 0.06647639150431685
# it 10000 : 0.05554585256277002

# FourierAnsatz(4, [[0,1,2,3],[0,2,1,3],[0,3,1,2]])
# qubit count: 4
# circuit depth: 10
# parameter count: 18
# sc 100 : 0.3410510824983197
# sc 300 : 0.2448850438076755
# sc 1000 : 0.20179895609771342
# it 100 : 0.3033924310827193
# it 300 : 0.1688650876767093
# it 1000 : 0.10465858077280084
# it 3000 : 0.0658912224717679
# it 10000 : 0.054644128739449474

# sc 100 time: 0.008081
# sc 300 time: 0.025463999999999997
# sc 1000 time: 0.08534399999999999
# it 100 time: 0.1098
# it 300 time: 0.34709999999999996
# it 1000 time: 1.1637
# it 3000 time: 3.515000000000001
# it 10000 time: 11.727500000000001

################# transpiler problem #################
# HardwareEfficientReal(qubit_count=4, reps=3)
# qubit count: 4
# parameter count: 16
# (common) init depth   = 11
# (it) transpiled depth = 25
# (it) converted depth  = 232
# (sc) transpiled depth = 44
# (sc) converted depth  = 47

if __name__ == "__main__":
    n_qubits = 4
    hamiltonian = prepare_problem(n_qubits)
    parametric_state_sc = prepare_ansatz("sc", n_qubits)
    parametric_state_it = prepare_ansatz("it", n_qubits)
    print("qubit count:", parametric_state_sc.parametric_circuit.qubit_count)
    print("parameter count:", parametric_state_sc.parametric_circuit.parameter_count)
    print("circuit depth (sc):", parametric_state_sc.parametric_circuit.depth)
    print("circuit depth (it):", parametric_state_it.parametric_circuit.depth)
    tester = SamplingTester(
        hamiltonian,
        parametric_state_sc,
        parametric_state_it,
        [10000],
        [],
        30
    )
    for _ in range(1):
        params = np.random.uniform(0, 2*np.pi, parametric_state_sc.parametric_circuit.parameter_count)
        tester.test(params.tolist())