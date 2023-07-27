import numpy as np
from quri_parts.core.operator import Operator
from quri_parts.core.estimator.sampling.estimator import _Estimate
from quri_parts.core.state import ParametricCircuitQuantumState
from quri_parts.core.estimator.sampling.pauli import general_pauli_sum_sample_variance
from quri_parts.core.sampling.shots_allocator import create_proportional_shots_allocator
from collections.abc import Mapping
from typing import Iterable, Union, cast, Tuple

import math
from collections.abc import Collection

from quri_parts.core.operator import CommutablePauliSet, Operator
from quri_parts.core.sampling import PauliSamplingSetting, PauliSamplingShotsAllocator

import numpy as np

from quri_parts.core.measurement import PauliReconstructorFactory
from quri_parts.core.operator import PAULI_IDENTITY, CommutablePauliSet, PauliLabel
from quri_parts.core.sampling import MeasurementCounts
from common import prepare_sampling_estimator
from GivensAnsatz import GivensAnsatz

def create_threshold_shots_allocator(remove_rate: float = 0.3) -> PauliSamplingShotsAllocator:
    def allocator(
        operator: Operator, pauli_sets: Collection[CommutablePauliSet], total_shots: int
    ) -> Collection[PauliSamplingSetting]:
        pauli_sets = tuple(pauli_sets)  # to fix the order of elements

        weights = [
            math.sqrt(sum([abs(operator[pauli_label]) ** 2 for pauli_label in pauli_set]))
            for pauli_set in pauli_sets
        ]
        ascend_order = np.argsort(weights)
        remove_amount = sum(weights) * remove_rate
        for idx in ascend_order:
            if remove_amount <= weights[idx]:
                break
            remove_amount -= weights[idx]
            weights[idx] = 0
        weight_sum = sum(weights)
        ratios = [w / weight_sum for w in weights]
        shots_list = [
            math.floor(total_shots * ratio) for ratio in ratios
        ]
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

class MeasurementResult:
    def __init__(self):
        self.coefs: dict[PauliLabel, complex] = {}
        self.num_measurements: dict[PauliLabel, list[int]] = {}
        self.expectation_values: dict[PauliLabel, list[float]] = {}
        self.measurement_results: dict[PauliLabel, list[complex]] = {}
        self.const: float = 0.0
    
    def reset(self):
        self.coefs.clear()
        self.num_measurements.clear()
        self.expectation_values.clear()
        self.measurement_results.clear()
    
    def add_result(
        self, pauli_label: PauliLabel, coef: complex,
        num_measurement: int, expectation_value: float, measurement_result: complex
    ):
        if(pauli_label not in self.coefs):
            self.coefs[pauli_label] = coef
            self.num_measurements[pauli_label] = []
            self.expectation_values[pauli_label] = []
            self.measurement_results[pauli_label] = []
        self.num_measurements[pauli_label].append(num_measurement)
        self.expectation_values[pauli_label].append(expectation_value)
        self.measurement_results[pauli_label].append(measurement_result)

def mitigated_pauli_expectation_estimator(
    counts: MeasurementCounts,
    pauli: PauliLabel,
    reconstructor_factory: PauliReconstructorFactory,
    bit_flip_rate: float
) -> float:
    if len(counts) == 0:
        raise ValueError("No measurement counts supplied (counts is empty).")

    if pauli == PAULI_IDENTITY:
        return 1.0

    reconstructor = reconstructor_factory(pauli)

    val = 0.0
    for key, count in counts.items():
        val += reconstructor(key) * count
    a = (val + sum(counts.values())) / 2
    b = sum(counts.values()) - a
    val /= sum(counts.values())

    n = str(pauli).count(" ") + 1

    # error mitigation for bit-flip noise
    _p = a / (a + b)
    e0, e1 = 1, 0
    for i in range(n):
        e0, e1 = e0 * (1 - bit_flip_rate) + e1 * bit_flip_rate, e0 * bit_flip_rate + e1 * (1 - bit_flip_rate)
    e = e1
    p = (_p - e) / (1 - e * 2)
    return p * 2 - 1

def mitigated_pauli_sum_expectation_estimator(
    counts: MeasurementCounts,
    pauli_set: CommutablePauliSet,
    coefs: Mapping[PauliLabel, complex],
    reconstructor_factory: PauliReconstructorFactory,
    measurement_result: MeasurementResult,
    bit_flip_rate: float
) -> complex:
    pauli_exp_and_coefs = []
    for pauli in pauli_set:
        if(pauli in coefs):
            pauli_exp_and_coefs.append(
                (
                    mitigated_pauli_expectation_estimator(counts, pauli, reconstructor_factory, bit_flip_rate),
                    coefs[pauli]
                )
            )
            measurement_result.add_result(
                pauli, coefs[pauli],
                math.floor(sum(counts.values()) + 0.5),
                mitigated_pauli_expectation_estimator(counts, pauli, reconstructor_factory, bit_flip_rate),
                mitigated_pauli_expectation_estimator(counts, pauli, reconstructor_factory, bit_flip_rate) * coefs[pauli]
            )
    if not pauli_exp_and_coefs:
        return 0
    pauli_exp_seq, coef_seq = zip(*pauli_exp_and_coefs)
    raw_energy = cast(complex, np.inner(pauli_exp_seq, coef_seq))
    return raw_energy

def get_mitigated_measurement_value(
    _op: Operator, const: complex,
    _pauli_sets: Iterable[CommutablePauliSet], _pauli_recs: Iterable[PauliReconstructorFactory],
    _sampling_counts: dict[CommutablePauliSet, MeasurementCounts],
    measurement_result: MeasurementResult,
    bit_flip_rate: float,
    depolarizing_rate: float
) -> Tuple[float, float]:
    val, var_sum = const, 0.0
    for pauli_set, pauli_rec in zip(
        _pauli_sets, _pauli_recs
    ):
        counts = _sampling_counts[pauli_set]
        val += mitigated_pauli_sum_expectation_estimator(
            counts, pauli_set, _op, pauli_rec, measurement_result, bit_flip_rate
        )
        var_sum += general_pauli_sum_sample_variance(
            counts, pauli_set, _op, pauli_rec
        ) / sum(counts.values())
    val_identity = _op.constant

    # Mitigating depolarizing noise (PRL, 2021)
    new_val = val_identity + (val - val_identity) / (1 - depolarizing_rate)
    new_var_sum = var_sum / (1 - depolarizing_rate) ** 2
    return new_val.real, np.sqrt(new_var_sum)

class Measure:
    def __init__(
        self, hardware_type: str,
        hamiltonian: Operator, ansatz: GivensAnsatz,
        total_shots: int,
        optimization_level: int = 0,
        bit_flip_error: float = 0,
        depolarizing_rate: float = 0
    ) -> None:
        self.hardware_type = hardware_type
        self.parameter_convert_func = ansatz.parameter_convert_func
        self.total_shots = total_shots
        self.hamiltonian = hamiltonian
        self.parametric_circuit = ParametricCircuitQuantumState(ansatz.qubit_count, ansatz._circuit)
        self.optimization_level = optimization_level
        self.bit_flip_error = bit_flip_error
        self.depolarizing_rate = depolarizing_rate
        self.measurement_result = MeasurementResult()
    
    def measure(self, params: list[float]) -> Tuple[float, float]:
        estimator = prepare_sampling_estimator(self.hardware_type, self.total_shots, create_proportional_shots_allocator())
        estimate: _Estimate = estimator(self.hamiltonian, self.parametric_circuit, [self.parameter_convert_func(params)])[0]
        cost, error = estimate.value.real, estimate.error

        merged_sampling_counts: dict[CommutablePauliSet, dict[int, Union[int, float]]] = {}
        for pauli_set, counts in zip(estimate._pauli_sets, estimate._sampling_counts):
            d = {}
            for k, v in counts.items(): d[k] = v
            merged_sampling_counts[pauli_set] = d
        
        self.measurement_result.const = estimate._const.real
        cost, error = get_mitigated_measurement_value(
            estimate._op, estimate._const,
            estimate._pauli_sets, estimate._pauli_recs, merged_sampling_counts,
            self.measurement_result,
            self.bit_flip_error,
            self.depolarizing_rate
        )
        return cost, error