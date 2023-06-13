from math import e
from git import Sequence
import numpy as np
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
from collections import defaultdict
from collections.abc import Mapping
from typing import Iterable, Union, cast, Tuple

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
from utils.challenge_2023 import ChallengeSampling

def mitigated_pauli_expectation_estimator(
    counts: MeasurementCounts,
    pauli: PauliLabel,
    reconstructor_factory: PauliReconstructorFactory,
    bit_flip_rate: float
) -> float:
    """An implementation of :class:`~PauliExpectationEstimator` for a given
    :class:`PauliReconstructorFactory`."""
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
    bit_flip_rate: float
) -> complex:
    """Estimate expectation value of a weighted sum of commutable Pauli
    operators from measurement counts and Pauli reconstructor.

    Note that this function calculates the sum for only Pauli operators
    contained in both of ``pauli_set`` and ``coefs``.
    """
    pauli_exp_and_coefs = [
        (
            mitigated_pauli_expectation_estimator(counts, pauli, reconstructor_factory, bit_flip_rate),
            coefs[pauli],
        )
        for pauli in pauli_set
        if pauli in coefs
    ]
    if not pauli_exp_and_coefs:
        return 0
    pauli_exp_seq, coef_seq = zip(*pauli_exp_and_coefs)
    return cast(complex, np.inner(pauli_exp_seq, coef_seq))

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

def get_measurement_info(
    _op: Operator,
    _pauli_sets: Iterable[CommutablePauliSet], _pauli_recs: Iterable[PauliReconstructorFactory],
    _sampling_counts: dict[CommutablePauliSet, MeasurementCounts]
) -> Tuple[dict[CommutablePauliSet, int], dict[CommutablePauliSet, float]]:
    count_dict, var_dict = {}, {}
    for pauli_set, pauli_rec in zip(
        _pauli_sets, _pauli_recs
    ):
        counts = _sampling_counts[pauli_set]
        count_dict[pauli_set] = sum(counts.values())
        est_var = estimate_variance_from_measurement(counts, pauli_set, _op, pauli_rec)
        var_dict[pauli_set] = est_var
    return count_dict, var_dict

def get_measurement_value(
    _op: Operator, const: complex,
    _pauli_sets: Iterable[CommutablePauliSet], _pauli_recs: Iterable[PauliReconstructorFactory],
    _sampling_counts: dict[CommutablePauliSet, MeasurementCounts]
) -> Tuple[float, float]:
    val, var_sum = const, 0.0
    for pauli_set, pauli_rec in zip(
        _pauli_sets, _pauli_recs
    ):
        counts = _sampling_counts[pauli_set]
        val += general_pauli_sum_expectation_estimator(
            counts, pauli_set, _op, pauli_rec
        )
        var_sum += general_pauli_sum_sample_variance(
            counts, pauli_set, _op, pauli_rec
        ) / sum(counts.values())
    return val.real, np.sqrt(var_sum)

def get_mitigated_measurement_value(
    _op: Operator, const: complex,
    _pauli_sets: Iterable[CommutablePauliSet], _pauli_recs: Iterable[PauliReconstructorFactory],
    _sampling_counts: dict[CommutablePauliSet, MeasurementCounts],
    bit_flip_rate: float
) -> Tuple[float, float]:
    val, var_sum = const, 0.0
    for pauli_set, pauli_rec in zip(
        _pauli_sets, _pauli_recs
    ):
        counts = _sampling_counts[pauli_set]
        val += mitigated_pauli_sum_expectation_estimator(
            counts, pauli_set, _op, pauli_rec, bit_flip_rate
        )
        var_sum += general_pauli_sum_sample_variance(
            counts, pauli_set, _op, pauli_rec
        ) / sum(counts.values())
    return val.real, np.sqrt(var_sum)

def create_variance_proportional_shots_allocator(
    count_dict: dict[CommutablePauliSet, int], var_dict: dict[CommutablePauliSet, float]
) -> PauliSamplingShotsAllocator:

    def allocator(
        operator: Operator, pauli_sets: Collection[CommutablePauliSet], total_shots: int
    ) -> Collection[PauliSamplingSetting]:
        pauli_sets = tuple(pauli_sets)  # to fix the order of elements
        count_array = [count_dict[pauli_set] if pauli_set in count_dict else 0 for pauli_set in pauli_sets]
        var_array = [var_dict[pauli_set] if pauli_set in var_dict else 0 for pauli_set in pauli_sets]
        var_sum = sum(var_array)

        ref_total_shots = total_shots + sum(count_array)
        while(True):
            shots_list = [max(0, int(ref_total_shots * var_array[i] / var_sum - count_array[i])) for i in range(len(count_array))]
            sum_shots = sum(shots_list)
            if(sum_shots == total_shots): break

            if(abs(sum_shots - total_shots) < 10):
                ref_total_shots += total_shots - sum_shots
            else:
                ref_total_shots *= total_shots / sum_shots

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

class Measure:
    def __init__(
        self, noise: bool, hardware_type: str, total_shots: int,
        hamiltonian: Operator, parametric_circuit: ParametricCircuitQuantumState,
        optimization_level: int = 0,
        bit_flip_error: float = 0
    ) -> None:
        self.noise = noise
        self.hardware_type = hardware_type
        self.total_shots = total_shots
        self.hamiltonian = hamiltonian
        self.parametric_circuit = parametric_circuit
        self.optimization_level = optimization_level
        self.bit_flip_error = bit_flip_error
    
    def measure(self, params: list[float]) -> Tuple[float, float]:
        prepare_estimator = prepare_sampling_estimator if self.noise else prepare_noiseless_estimator

        if(self.optimization_level == 0):
            estimator = prepare_estimator(self.hardware_type, self.total_shots)
            estimate: _Estimate = estimator(self.hamiltonian, self.parametric_circuit, [params])[0]
            cost, error = estimate.value.real, estimate.error

            merged_sampling_counts: dict[CommutablePauliSet, dict[int, Union[int, float]]] = {}
            for pauli_set, counts in zip(estimate._pauli_sets, estimate._sampling_counts):
                d = {}
                for k, v in counts.items(): d[k] = v
                merged_sampling_counts[pauli_set] = d
            cost, error = get_measurement_value(
                estimate._op, estimate._const,
                estimate._pauli_sets, estimate._pauli_recs, merged_sampling_counts
            )
            cost, error = get_mitigated_measurement_value(
                estimate._op, estimate._const,
                estimate._pauli_sets, estimate._pauli_recs, merged_sampling_counts,
                self.bit_flip_error
            )
            return cost, error
        
        elif(self.optimization_level == 1):
            merged_sampling_counts: dict[CommutablePauliSet, dict[int, Union[int, float]]] = {}

            allocator_1 = create_equipartition_shots_allocator()
            estimator_1 = prepare_estimator(self.hardware_type, 50, allocator_1)
            estimate_1: _Estimate = estimator_1(self.hamiltonian, self.parametric_circuit, [params])[0]
            for pauli_set, counts in zip(estimate_1._pauli_sets, estimate_1._sampling_counts):
                d = {}
                for k, v in counts.items(): d[k] = v
                merged_sampling_counts[pauli_set] = d
            count_dict_1, var_dict_1 = get_measurement_info(
                estimate_1._op,
                estimate_1._pauli_sets, estimate_1._pauli_recs, merged_sampling_counts
            )
            
            allocator_2 = create_variance_proportional_shots_allocator(count_dict_1, var_dict_1)
            estimator_2 = prepare_estimator(self.hardware_type, self.total_shots - sum(count_dict_1.values()), allocator_2)
            estimate_2: _Estimate = estimator_2(self.hamiltonian, self.parametric_circuit, [params])[0]
            for pauli_set, counts in zip(estimate_2._pauli_sets, estimate_2._sampling_counts):
                for k, v in counts.items():
                    if(k in merged_sampling_counts[pauli_set]): merged_sampling_counts[pauli_set][k] += v
                    else: merged_sampling_counts[pauli_set][k] = v
            
            return get_measurement_value(
                estimate_2._op, estimate_2._const,
                estimate_2._pauli_sets, estimate_2._pauli_recs, merged_sampling_counts
            )
        
        elif(self.optimization_level == 2):
            merged_sampling_counts: dict[CommutablePauliSet, dict[int, Union[int, float]]] = {}
            remaining_shots = self.total_shots
            # print(remaining_shots)

            allocator_1 = create_equipartition_shots_allocator()
            estimator_1 = prepare_estimator(self.hardware_type, 50, allocator_1)
            estimate_1: _Estimate = estimator_1(self.hamiltonian, self.parametric_circuit, [params])[0]
            for pauli_set, counts in zip(estimate_1._pauli_sets, estimate_1._sampling_counts):
                d = {}
                for k, v in counts.items(): d[k] = v
                merged_sampling_counts[pauli_set] = d
            count_dict_1, var_dict_1 = get_measurement_info(
                estimate_1._op,
                estimate_1._pauli_sets, estimate_1._pauli_recs, merged_sampling_counts
            )
            # print(list(count_dict_1.values()))
            remaining_shots -= sum([sum(counts.values()) for counts in estimate_1._sampling_counts])
            # print(remaining_shots)
            # print([list(counts.values()) for counts in merged_sampling_counts.values()])
            
            allocator_2 = create_proportional_shots_allocator()
            estimator_2 = prepare_estimator(self.hardware_type, int(remaining_shots * 0.7), allocator_2)
            estimate_2: _Estimate = estimator_2(self.hamiltonian, self.parametric_circuit, [params])[0]
            for pauli_set, counts in zip(estimate_2._pauli_sets, estimate_2._sampling_counts):
                for k, v in counts.items():
                    if(k in merged_sampling_counts[pauli_set]): merged_sampling_counts[pauli_set][k] += v
                    else: merged_sampling_counts[pauli_set][k] = v
            count_dict_2, var_dict_2 = get_measurement_info(
                estimate_2._op,
                estimate_2._pauli_sets, estimate_2._pauli_recs, merged_sampling_counts
            )
            # print(list(count_dict_2.values()))
            remaining_shots -= sum([sum(counts.values()) for counts in estimate_2._sampling_counts])
            # print(remaining_shots)
            # print([list(counts.values()) for counts in merged_sampling_counts.values()])

            allocator_3 = create_variance_proportional_shots_allocator(count_dict_2, var_dict_2)
            estimator_3 = prepare_estimator(self.hardware_type, remaining_shots, allocator_3)
            estimate_3: _Estimate = estimator_3(self.hamiltonian, self.parametric_circuit, [params])[0]
            for pauli_set, counts in zip(estimate_3._pauli_sets, estimate_3._sampling_counts):
                for k, v in counts.items():
                    if(k in merged_sampling_counts[pauli_set]): merged_sampling_counts[pauli_set][k] += v
                    else: merged_sampling_counts[pauli_set][k] = v
            count_dict_3, var_dict_3 = get_measurement_info(
                estimate_3._op,
                estimate_3._pauli_sets, estimate_3._pauli_recs, merged_sampling_counts
            )
            # print(list(count_dict_3.values()))
            remaining_shots -= sum([sum(counts.values()) for counts in estimate_3._sampling_counts])
            # print(remaining_shots)
            # exit()
            return get_measurement_value(
                estimate_3._op, estimate_3._const,
                estimate_3._pauli_sets, estimate_3._pauli_recs, merged_sampling_counts
            )
        raise ValueError("Invalid optimization level")