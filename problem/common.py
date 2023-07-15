import sys
from typing import Callable

from FourierAnsatz import FourierAnsatz
from problem.GivensAnsatz import GivensAnsatz, GivensAnsatz_it_4, GivensAnsatz_it_8
sys.path.append("../")

from openfermion.transforms import jordan_wigner
from openfermion.utils import load_operator
from quri_parts.algo.ansatz import HardwareEfficientReal
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement
from quri_parts.core.estimator.sampling.estimator import _Estimate
from quri_parts.core.operator import Operator
from quri_parts.core.sampling.shots_allocator import (
    create_proportional_shots_allocator,
    create_equipartition_shots_allocator
)
from quri_parts.core.state import ParametricCircuitQuantumState, ComputationalBasisState
from quri_parts.openfermion.operator import operator_from_openfermion_op
from quri_parts.algo.mitigation.zne import (
    create_folding_random,
    scaling_circuit_folding,
    create_folding_left,
    create_polynomial_extrapolate,
    create_exp_extrapolate,
    create_zne_estimator
)

from utils.challenge_2023 import ChallengeSampling, TimeExceededError
challenge_sampling = ChallengeSampling(noise=True)
noiseless_sampling = ChallengeSampling(noise=False)

from FourierAnsatz import FourierAnsatz_4, FourierAnsatz_8

def prepare_problem(n_qubits: int, sample_idx : int = 0) -> Operator:
    if(n_qubits not in [4, 8]):
        print("n_qubits must be 4 or 8")
        exit()
    
    # problem setting
    filename = f"{n_qubits}_qubits_H" if sample_idx == 0 else f"hamiltonian_samples/{n_qubits}_qubits_H_{sample_idx}"
    ham = load_operator(
        file_name=filename,
        data_directory="../hamiltonian",
        plain_text=False,
    )
    jw_hamiltonian = jordan_wigner(ham)
    hamiltonian = operator_from_openfermion_op(jw_hamiltonian)

    # for pauli, coef in hamiltonian.items():
    #     S = str(pauli).split(" ")
    #     p = ["I"] * n_qubits
    #     for s in S:
    #         if(s[0] == "I"): continue
    #         p[int(s[1])] = s[0]
    #     print("".join(p), f"{coef.real: .12f}")
    # exit()
    return hamiltonian

# def prepare_ansatz(hardware_type: str, n_qubits: int) -> FourierAnsatz:
#     n_site = n_qubits // 2
#     # hw_ansatz = HardwareEfficientReal(qubit_count=n_qubits, reps=3)
#     if(n_qubits == 4): hw_ansatz = FourierAnsatz_4(hardware_type)
#     elif(n_qubits == 8): hw_ansatz = FourierAnsatz_8(hardware_type)
#     else:
#         print("n_qubits must be 4 or 8")
#         exit()
#     return hw_ansatz

# def prepare_ansatz(hardware_type: str, n_qubits: int) -> GivensAnsatz:
#     if(hardware_type == "sc"):
#         print("hardware_type must be 'it'")
#         exit()
#     if(n_qubits == 4): ansatz = GivensAnsatz_it_4()
#     elif(n_qubits == 8): ansatz = GivensAnsatz_it_8()
#     else:
#         print("n_qubits must be 4 or 8")
#         exit()
#     return ansatz

def prepare_sampling_estimator(
    hardware_type: str, num_shots: int,
    shots_allocator = create_proportional_shots_allocator()
):
    sampling_estimator = challenge_sampling.create_concurrent_parametric_sampling_estimator(
        total_shots = num_shots,
        measurement_factory = bitwise_commuting_pauli_measurement,
        shots_allocator = shots_allocator,
        hardware_type = hardware_type,
    )
    return sampling_estimator

def prepare_noiseless_estimator(
    hardware_type: str,
    num_shots: int,
    shots_allocator = create_proportional_shots_allocator()
):
    sampling_estimator = noiseless_sampling.create_concurrent_parametric_sampling_estimator(
        total_shots = num_shots,
        measurement_factory = bitwise_commuting_pauli_measurement,
        shots_allocator = shots_allocator,
        hardware_type = hardware_type,
    )
    return sampling_estimator

def prepare_zne_estimator(hardware_type, num_shots):
    scale_factors = [1, 1, 1, 1.5, 1.5, 2]
    # choose an extrapolation method
    extrapolate_method = create_polynomial_extrapolate(order=1)
    # choose how folding your circuit
    folding_method = create_folding_random()

    concurrent_estimator = challenge_sampling.create_concurrent_sampling_estimator(
        num_shots,
        measurement_factory = bitwise_commuting_pauli_measurement,
        shots_allocator = create_proportional_shots_allocator(),
        hardware_type = hardware_type
    )

    # construct estimator by using zne (only concurrent estimator can be used)
    zne_estimator = create_zne_estimator(
        concurrent_estimator, scale_factors, extrapolate_method, folding_method
    )
    # # by using this estimator, you can obtain an estimated value with ZNE
    # zne_estimated_value = zne_estimator(op, circuit_state)

    # print(f"zne_estimated_value :{zne_estimated_value.value} ")





    # poly_extrapolation = create_polynomial_extrapolate(order = len(scale_factors) - 1)
    # random_folding = create_folding_random()
    # zne_estimator = create_zne_estimator(
    #     estimator, scale_factors, poly_extrapolation, random_folding
    # )
    return zne_estimator

class CostEvaluator:
    def __init__(
        self, name, hardware_type,
        get_estimate: Callable[[str, str, list[float]], _Estimate]
    ) -> None:
        self.name = name
        self.hardware_type = hardware_type
        self.get_estimate = get_estimate
        self.reset()
    
    def reset(self):
        self.min_cost = 100000000
        self.min_params = None

    def eval(self, params, time_limit: float = 1000, reset: bool = False, show_output: bool = False):
        if(challenge_sampling.total_quantum_circuit_time > time_limit):
            raise TimeExceededError(challenge_sampling.total_quantum_circuit_time, 0)
        if(reset):
            self.reset()
        
        self.latest_estimate = self.get_estimate(self.name, self.hardware_type, params)
        cost, error = self.latest_estimate.value.real, self.latest_estimate.error
        if cost < self.min_cost:
            self.min_cost = cost
            self.min_params = params
            if(show_output):
                print("params updated:", cost, "+-", error)
        else:
            if(show_output):
                print(cost, "+-", error)
        return cost, error

if __name__ == "__main__":
    pass