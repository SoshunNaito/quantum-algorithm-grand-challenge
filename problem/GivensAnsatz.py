from typing import Union, Callable
import numpy as np
from scipy.optimize import differential_evolution
import random
from quri_parts.circuit import (
    ImmutableLinearMappedUnboundParametricQuantumCircuit,
    ImmutableBoundParametricQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuit,
)
from quri_parts.algo.optimizer import SPSA, OptimizerStatus
from quri_parts.circuit.gate import ParametricQuantumGate, QuantumGate
from quri_parts.circuit.parameter import CONST, Parameter
from quri_parts.circuit.parameter_mapping import LinearParameterMapping, ParameterOrLinearFunction
from quri_parts.circuit.parameter_mapping import ParameterOrLinearFunction
from quri_parts.quantinuum.circuit import RZZ, ZZ, U1q

from BaseAnsatz import BaseAnsatz
from CosineSum.CosineSumGenerator import GenerateCosineSumInstance
from CosineSum.CosineSumSolver_CG import CosineSumSolver_CG

class BaseLayer:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.num_parameters = 0
        self.num_circuit_parameters = 0
        self.num_single_qubit_gates = 0
        self.num_multi_qubit_gates = 0
        self.parameter_blocks = []
    
    def parameter_convert_func(self, params: list[float]) -> list[float]:
        raise NotImplementedError
    
    def add_gates(self, circuit: LinearMappedUnboundParametricQuantumCircuit):
        raise NotImplementedError

class InitialStateLayer(BaseLayer):
    def __init__(self, n_qubits: int, configure: str):
        super().__init__(n_qubits)
        self.configure = configure
    
    def parameter_convert_func(self, params: list[float]) -> list[float]:
        return params

    def _approx_GHZ_state(
        self, circuit: LinearMappedUnboundParametricQuantumCircuit,
        a: int, b : int
    ):
        circuit.add_gate(U1q(a, np.pi / 2, -np.pi / 2))
        circuit.add_gate(U1q(b, np.pi / 2, np.pi / 2))
        circuit.add_gate(ZZ(a, b))
        circuit.add_gate(U1q(a, np.pi / 2, np.pi / 2))
        circuit.add_gate(U1q(a, np.pi / 2, np.pi / 2))
    
    def _approx_CX_gate(
        self, circuit: LinearMappedUnboundParametricQuantumCircuit,
        a: int, b : int
    ):
        circuit.add_gate(U1q(b, np.pi / 2, np.pi))
        circuit.add_gate(RZZ(a, b, -np.pi / 2))
        circuit.add_gate(U1q(b, np.pi / 2, np.pi / 2))
    
    def _approx_NCX_gate(
        self, circuit: LinearMappedUnboundParametricQuantumCircuit,
        a: int, b : int
    ):
        circuit.add_gate(U1q(b, np.pi / 2, np.pi))
        circuit.add_gate(RZZ(a, b, np.pi / 2))
        circuit.add_gate(U1q(b, np.pi / 2, np.pi / 2))
    
    def add_gates(self, circuit: LinearMappedUnboundParametricQuantumCircuit):
        super().__init__(self.n_qubits)
        n_zeros = 0
        n_ones = 0
        symbols: list[int] = []
        for i in range(self.n_qubits):
            if(self.configure[i] == "0"):
                n_zeros += 1
            elif(self.configure[i] == "1"):
                circuit.add_gate(U1q(i, np.pi, np.pi / 2))
                n_ones += 1
            elif(self.configure[i] == "s"):
                symbols.append(i)
            else:
                raise ValueError("Invalid configure string")
        if(self.n_qubits // 2 - n_ones == 1):
            if(len(symbols) == 1):
                for i in range(self.n_qubits):
                    if(self.configure[i] == "s"):
                        circuit.add_gate(U1q(i, np.pi, np.pi / 2))
            elif(len(symbols) == 2):
                a, b = symbols[0], symbols[1]
                self._approx_GHZ_state(circuit, a, b)
            else:
                raise NotImplementedError
        if(self.n_qubits // 2 - n_ones == 2):
            if(len(symbols) == 4):
                a, b, c, d = symbols[0], symbols[1], symbols[2], symbols[3]
                self._approx_GHZ_state(circuit, a, c)
                self._approx_CX_gate(circuit, a, b)
                self._approx_CX_gate(circuit, c, d)
            else:
                raise NotImplementedError
        if(n_zeros == 0 and n_ones == 0):
            if(len(symbols) == 4):
                self._approx_GHZ_state(circuit, 0, 2)
                self._approx_CX_gate(circuit, 0, 1)
                self._approx_CX_gate(circuit, 2, 3)
            elif(len(symbols) == 8):
                self._approx_GHZ_state(circuit, 0, 4)
                self._approx_CX_gate(circuit, 0, 1)
                self._approx_CX_gate(circuit, 0, 2)
                self._approx_CX_gate(circuit, 1, 3)
                self._approx_CX_gate(circuit, 4, 5)
                self._approx_CX_gate(circuit, 4, 6)
                self._approx_CX_gate(circuit, 5, 7)
            else:
                raise NotImplementedError

class GivensLayer(BaseLayer):
    def __init__(self, n_qubits: int, layers: list[list[int]], initial_state: Union[str, None] = None):
        super().__init__(n_qubits)
        self.layers = layers
        self.initial_state = initial_state
    
    def parameter_convert_func(self, params: list[float]) -> list[float]:
        ans = []
        assert(len(params) == self.num_parameters)
        for param in params:
            ans += [param / 2, -param / 2]
        return ans

    def _simplified_givens_rotation(
        self, circuit: LinearMappedUnboundParametricQuantumCircuit,
        a: int, b: int, theta_plus: Parameter, theta_minus: Parameter
    ):
        circuit.add_gate(RZZ(a, b, np.pi / 2))
        circuit.add_gate(U1q(a, np.pi / 2, np.pi / 2))
        circuit.add_gate(U1q(b, np.pi / 2, np.pi / 2))
        circuit.add_ParametricRZ_gate(a, theta_plus)
        circuit.add_ParametricRZ_gate(b, theta_minus)
        circuit.add_gate(U1q(a, np.pi / 2, -np.pi / 2))
        circuit.add_gate(U1q(b, np.pi / 2, -np.pi / 2))
        circuit.add_gate(RZZ(a, b, -np.pi / 2))
        
    def add_gates(self, circuit: LinearMappedUnboundParametricQuantumCircuit):
        super().__init__(self.n_qubits)
        if(self.initial_state is not None):
            for i in range(self.n_qubits):
                # prepare Hartree-Fock state
                circuit.add_gate(U1q(i, np.pi / 2, -np.pi / 2 if self.initial_state[i] == "0" else np.pi / 2))
                self.num_single_qubit_gates += 1
        else:
            for i in range(self.n_qubits):
                circuit.add_gate(U1q(i, np.pi / 2, 0))
                circuit.add_RZ_gate(i, -np.pi / 2)
                self.num_single_qubit_gates += 2
        for layer_idx, layer in enumerate(self.layers):
            for j in range(1, len(layer), 2):
                i = j - 1
                theta_plus, theta_minus = circuit.add_parameters(
                    f"theta_givens_{layer_idx}_{i//2}_plus", f"theta_givens_{layer_idx}_{i//2}_minus"
                )
                self.num_parameters += 1
                self.num_circuit_parameters += 2
                self.parameter_blocks.append(1)

                qi, qj = layer[i], layer[j]
                self._simplified_givens_rotation(circuit, qi, qj, theta_plus, theta_minus)
                self.num_single_qubit_gates += 6
                self.num_multi_qubit_gates += 2
        for i in range(self.n_qubits):
            circuit.add_RZ_gate(i, np.pi / 2)
            circuit.add_gate(U1q(i, np.pi / 2, np.pi))
            self.num_single_qubit_gates += 2

class PhaseRotationLayer(BaseLayer):
    def __init__(self, n_qubits: int, layers: list[list[int]]):
        super().__init__(n_qubits)
        self.layers = layers
    
    def parameter_convert_func(self, params: list[float]) -> list[float]:
        ans = []
        assert(len(params) == self.num_parameters)
        idx = 0
        for cnt in self.parameter_blocks:
            if(cnt == 3):
                i, j, k = idx, idx + 1, idx + 2
                si, sj, sk = params[i], params[j], params[k]
                s = (si + sj + sk) / 2
                ans += [s - si, s - sj, s - sk]
            elif(cnt == 1):
                ans += [params[idx]]
            idx += cnt
        return ans

    def add_gates(self, circuit: LinearMappedUnboundParametricQuantumCircuit):
        super().__init__(self.n_qubits)
        for layer_idx, layer in enumerate(self.layers):
            if(layer_idx == 0):
                for j in range(1, len(layer), 2):
                    i = j - 1
                    theta_1, theta_2, theta_3 = circuit.add_parameters(
                        f"theta_{layer_idx}_{i//2}_1", f"theta_{layer_idx}_{i//2}_2", f"theta_{layer_idx}_{i//2}_3"
                    )
                    self.num_parameters += 3
                    self.num_circuit_parameters += 3
                    self.parameter_blocks.append(3)

                    # diag([0, t2+t3, t1+t3, t1+t2])
                    qi, qj = layer[i], layer[j]
                    circuit.add_ParametricRZ_gate(qi, theta_1)
                    circuit.add_ParametricRZ_gate(qj, theta_2)
                    circuit.add_ParametricPauliRotation_gate([qi, qj], [3, 3], theta_3)

                    self.num_single_qubit_gates += 2
                    self.num_multi_qubit_gates += 1
            else:
                for j in range(1, len(layer), 2):
                    i = j - 1
                    theta = circuit.add_parameter(f"theta_{layer_idx}_{i//2}")
                    self.num_parameters += 1
                    self.num_circuit_parameters += 1
                    self.parameter_blocks.append(1)

                    # diag([0, t, t, 0])
                    qi, qj = layer[i], layer[j]
                    circuit.add_ParametricPauliRotation_gate([qi, qj], [3, 3], theta)

                    self.num_single_qubit_gates += 0
                    self.num_multi_qubit_gates += 1

class AdditionalLayer(BaseLayer):
    def __init__(self, n_qubits: int, configure_layers: list[str]):
        super().__init__(n_qubits)
        self.configure_layers = configure_layers
    
    def parameter_convert_func(self, params: list[float]) -> list[float]:
        assert(len(params) == self.num_parameters)
        return params
    
    def add_gates(self, circuit: LinearMappedUnboundParametricQuantumCircuit):
        super().__init__(self.n_qubits)
        for layer_idx, configure in enumerate(self.configure_layers):
            for i in range(len(configure)):
                if(configure[i] == "I"):
                    continue
                elif(configure[i] == "X"):
                    theta = circuit.add_parameter(f"theta_{layer_idx}_X_{i//2}")
                    self.num_parameters += 1
                    self.num_circuit_parameters += 1
                    self.parameter_blocks.append(1)

                    circuit.add_gate(U1q(i, np.pi / 2, np.pi / 2))
                    circuit.add_ParametricRZ_gate(i, theta)
                    circuit.add_gate(U1q(i, np.pi / 2, -np.pi / 2))
                    self.num_single_qubit_gates += 3

                elif(configure[i] == "Y"):
                    theta = circuit.add_parameter(f"theta_{layer_idx}_Y_{i//2}")
                    self.num_parameters += 1
                    self.num_circuit_parameters += 1
                    self.parameter_blocks.append(1)

                    circuit.add_gate(U1q(i, np.pi / 2, np.pi))
                    circuit.add_ParametricRZ_gate(i, theta)
                    circuit.add_gate(U1q(i, np.pi / 2, 0))
                    self.num_single_qubit_gates += 3

                elif(configure[i] == "Z"):
                    theta = circuit.add_parameter(f"theta_{layer_idx}_Z_{i//2}")
                    self.num_parameters += 1
                    self.num_circuit_parameters += 1
                    self.parameter_blocks.append(1)

                    circuit.add_ParametricRZ_gate(i, theta)
                    self.num_single_qubit_gates += 1

class GivensAnsatz(ImmutableLinearMappedUnboundParametricQuantumCircuit, BaseAnsatz):
    def __init__(
        self, n_qubits: int,
        layers: list[BaseLayer]
    ):
        self.n_qubits = n_qubits
        self.layers = layers
        circuit = LinearMappedUnboundParametricQuantumCircuit(n_qubits)
        for layer in layers: layer.add_gates(circuit)

        self.num_single_qubit_gates = sum(layer.num_single_qubit_gates for layer in layers)
        self.num_multi_qubit_gates = sum(layer.num_multi_qubit_gates for layer in layers)
        self.num_parameters = sum(layer.num_parameters for layer in layers)
        self.num_additional_parameters = sum(layer.num_parameters if isinstance(layer, AdditionalLayer) else 0 for layer in layers)
        self.num_circuit_parameters = sum(layer.num_circuit_parameters for layer in layers)
        self.parameter_blocks = []
        for layer in layers: self.parameter_blocks += layer.parameter_blocks
        
        super().__init__(circuit)

    def parameter_convert_func(self, params: list[float]) -> list[float]:
        ans = []
        assert(len(params) == self.num_parameters)
        idx = 0
        for layer in self.layers:
            ans += layer.parameter_convert_func(params[idx : idx + layer.num_parameters])
            idx += layer.num_parameters
        return ans

class GivensAnsatz_it_4(GivensAnsatz):
    def __init__(self):
        super().__init__(4, [
            GivensLayer(4, [[0, 2, 1, 3], [0, 1, 2, 3], [0, 3, 1, 2]], initial_state = "1100"),
            PhaseRotationLayer(4, [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2]])
        ])

class GivensAnsatz_it_8(GivensAnsatz):
    def __init__(self, freeze_ones: list[int], freeze_zeros: list[int]):
        variables = []
        for i in range(8):
            if(i not in freeze_ones and i not in freeze_zeros):
                variables.append(i)

        if(len(variables) == 4 and len(freeze_ones) == 2):
            a, b, c, d = variables
            initial_states = "1100"
            layers = [
                [a, c, b, d],
                [a, d, b, c],
                [a, b, c, d],
            ]
        elif(len(variables) == 6 and len(freeze_ones) == 2):
            a, b, c, d, e, f = variables
            initial_states = "110000"
            layers = [
                [a, c, b, d],
                [a, d, b, e, c, f],
                [a, b, c, d, e, f],
            ]
        elif(len(variables) == 6 and len(freeze_ones) == 0):
            a, b, c, d, e, f = variables
            initial_states = "111100"
            layers = [
                [a, e, b, f],
                [a, d, b, e, c, f],
                [a, b, c, d, e, f],
            ]
        elif(len(variables) == 8 and len(freeze_ones) == 0):
            initial_states = "11110000"
            layers = [
                [0, 4, 1, 5, 2, 6, 3, 7],
                [0, 1, 2, 3, 4, 6, 5, 7],
                [0, 3, 1, 2, 4, 5, 6, 7],
                [0, 7, 1, 6, 2, 5, 3, 4],
            ]
        else:
            raise NotImplementedError()
        
        initial_state_config = ""
        for i in range(8):
            if(i in freeze_ones):
                initial_state_config += "1"
            elif(i in freeze_zeros):
                initial_state_config += "0"
            else:
                initial_state_config += initial_states[0]
                initial_states = initial_states[1:]

        super().__init__(8, [
            # InitialStateLayer(8, initial_state_config),
            GivensLayer(8, layers, initial_state_config),
            PhaseRotationLayer(8, layers),
        ])

class GivensAnsatz_it_8_additional(GivensAnsatz):
    def __init__(self, base_ansatz: GivensAnsatz_it_8, additional_layer_config: list[str]):
        layers: list[BaseLayer] = [AdditionalLayer(8, [additional_layer_config[0]])]
        for idx, layer in enumerate(base_ansatz.layers):
            layers.append(layer)
            layers.append(AdditionalLayer(8, [additional_layer_config[(idx + 1) % len(additional_layer_config)]]))
        super().__init__(8, layers)

class GivensAnsatzOptimizer:
    def __init__(self, ansatz: GivensAnsatz):
        self.ansatz = ansatz
    
    def optimize(
        self, measurement_func: Callable[[list[float]], float], init_params: list[float],
        num_iterations: int = 3, num_additional_measure_1 = 2, num_additional_measure_3 = 7
    ):
        current_params = init_params
        for itr in range(num_iterations):
            num_parameters = len(self.ansatz.parameter_blocks)
            block_indices = list(range(num_parameters))
            for block_idx in block_indices:
                cnt = self.ansatz.parameter_blocks[block_idx]
                idx = sum(self.ansatz.parameter_blocks[: block_idx])
                def measure(params: list[float]) -> float:
                    val = measurement_func(current_params[: idx] + params + current_params[idx + cnt :])
                    # print(f" {val}")
                    return val
                cosineSumInstance = GenerateCosineSumInstance(
                    cnt, True, measure,
                    num_additional_measure_1 if cnt == 1 else num_additional_measure_3
                )
                optimal_params = CosineSumSolver_CG(cosineSumInstance).solve(
                    [
                        (np.random.rand(cnt) * 2 * np.pi).tolist()
                        for _ in range(100)
                    ], True
                )
                print(f"{cosineSumInstance.eval(optimal_params)} ({measurement_func(current_params[: idx] + optimal_params + current_params[idx + cnt :])}) :", current_params[idx : idx + cnt], "->", optimal_params)
                current_params[idx : idx + cnt] = optimal_params
        return current_params
    
    def optimize_differential_evolution(self, measurement_func: Callable[[list[float]], float], init_params: list[float]):
        try:
            differential_evolution(
                lambda params: measurement_func(params.tolist()),
                [(p - np.pi / 4, p + np.pi / 4) for p in init_params],
            )
        except Exception as e: pass