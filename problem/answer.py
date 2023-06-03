import sys
sys.path.append("../")

import numpy as np
from scipy.optimize import minimize
from typing import Any

from openfermion.transforms import jordan_wigner
from openfermion.utils import load_operator
from quri_parts.algo.ansatz import HardwareEfficientReal
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement
from quri_parts.core.sampling.shots_allocator import (
    create_proportional_shots_allocator
)
from quri_parts.core.state import ParametricCircuitQuantumState, ComputationalBasisState
from quri_parts.openfermion.operator import operator_from_openfermion_op

from utils.challenge_2023 import ChallengeSampling, QuantumCircuitTimeExceededError
challenge_sampling = ChallengeSampling(noise=True)

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

        # problem setting
        n_site = 2
        n_qubits = 2 * n_site
        ham = load_operator(
            file_name=f"{n_qubits}_qubits_H",
            data_directory="../hamiltonian",
            plain_text=False,
        )
        jw_hamiltonian = jordan_wigner(ham)
        hamiltonian = operator_from_openfermion_op(jw_hamiltonian)

        # prepare ansatz
        hf_gates = ComputationalBasisState(n_qubits, bits=2**n_site-1).circuit.gates
        hw_ansatz = HardwareEfficientReal(qubit_count=n_qubits, reps=2)
        hw_hf = hw_ansatz.combine(hf_gates)
        parametric_state = ParametricCircuitQuantumState(n_qubits, hw_hf)

        # parameter settings for measurement
        hardware, shots_approx, shots_precise = "sc", 1000, 10000
        # hardware, shots_approx, shots_precise = "it", 100, 3000
        sampling_estimator_approx = challenge_sampling.create_concurrent_parametric_sampling_estimator(
            total_shots = shots_approx,
            measurement_factory = bitwise_commuting_pauli_measurement,
            shots_allocator = create_proportional_shots_allocator(),
            hardware_type = hardware
        )
        sampling_estimator_precise = challenge_sampling.create_concurrent_parametric_sampling_estimator(
            total_shots = shots_precise,
            measurement_factory = bitwise_commuting_pauli_measurement,
            shots_allocator = create_proportional_shots_allocator(),
            hardware_type = hardware
        )

        tmp_cost, tmp_params = 0, np.zeros(hw_ansatz.parameter_count)
        def eval(params, estimator, time_limit: float):
            if(challenge_sampling.total_quantum_circuit_time > time_limit):
                raise QuantumCircuitTimeExceededError(challenge_sampling.total_quantum_circuit_time)
            
            nonlocal tmp_cost, tmp_params
            cost = estimator(hamiltonian, parametric_state, [params])[0].value.real
            if cost < tmp_cost:
                tmp_cost = cost
                tmp_params = params
                print("params updated:", tmp_cost, tmp_params)
            return cost

        # optimize
        n_init_params = 10
        ans_cost, ans_params = 0, np.zeros(hw_ansatz.parameter_count)
        for itr in range(n_init_params):
            print("iteration:", itr)
            remaining_time = 1000 - challenge_sampling.total_quantum_circuit_time
            allocated_time = remaining_time / (n_init_params - itr)
            time_end_approx = challenge_sampling.total_quantum_circuit_time + allocated_time * 0.3
            time_end_precise = challenge_sampling.total_quantum_circuit_time + allocated_time

            init_params = np.random.rand(hw_ansatz.parameter_count) * 2 * np.pi
            print("<< approx >>")
            try:
                tmp_cost, tmp_params = 0, np.zeros(hw_ansatz.parameter_count)
                result = minimize(
                    lambda params: eval(params, sampling_estimator_approx, time_end_approx),
                    init_params,
                    method = "Powell",
                )
            except QuantumCircuitTimeExceededError: pass
            init_params = tmp_params
            
            print("<< precise >>")
            try:
                tmp_cost, tmp_params = 0, np.zeros(hw_ansatz.parameter_count)
                result = minimize(
                    lambda params: eval(params, sampling_estimator_precise, time_end_precise),
                    init_params,
                    method = "Powell",
                )
            except QuantumCircuitTimeExceededError: pass
            if(tmp_cost < ans_cost):
                ans_cost = tmp_cost
                ans_params = tmp_params
            print()
        return ans_cost


if __name__ == "__main__":
    run_algorithm = RunAlgorithm()
    print(run_algorithm.get_result())
