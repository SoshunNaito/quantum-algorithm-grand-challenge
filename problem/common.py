import sys
sys.path.append("../")

from openfermion.transforms import jordan_wigner
from openfermion.utils import load_operator
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement
from quri_parts.core.operator import Operator
from quri_parts.core.sampling.shots_allocator import create_proportional_shots_allocator
from quri_parts.openfermion.operator import operator_from_openfermion_op

from utils.challenge_2023 import ChallengeSampling
challenge_sampling = ChallengeSampling(noise=True)

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
    return hamiltonian

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