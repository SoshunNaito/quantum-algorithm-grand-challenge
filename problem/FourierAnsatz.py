import numpy as np
from quri_parts.circuit import (
    ImmutableLinearMappedUnboundParametricQuantumCircuit,
    LinearMappedUnboundParametricQuantumCircuit,
)
from quri_parts.circuit.gate import ParametricQuantumGate, QuantumGate
from quri_parts.circuit.parameter import CONST, Parameter
from quri_parts.circuit.parameter_mapping import LinearParameterMapping, ParameterOrLinearFunction
from quri_parts.circuit.parameter_mapping import ParameterOrLinearFunction
from quri_parts.quantinuum.circuit import RZZ, ZZ, U1q

# def add_ParametricRZZ_gate(self, qubit_index: int) -> Parameter:
#     if qubit_index >= self.qubit_count:
#         raise ValueError(
#             "The indices of the gate applied must be smaller than qubit_count"
#         )
#     """Add a parametric RZ gate to the circuit."""
#     p = Parameter()
#     self._gates.append((ParametricRZ(qubit_index), p))
#     return p

# def add_ParametricRZZ_gate(
#     circuit: LinearMappedUnboundParametricQuantumCircuit, qubit_index: int, angle: ParameterOrLinearFunction
# ) -> None:
#     """Add a parametric RZ gate to the circuit."""
#     circuit._check_param_exist(angle)
#     param = circuit._circuit.add_ParametricRZZ_gate(qubit_index)
#     circuit._param_mapping = circuit._param_mapping.with_data_updated(
#         out_params_addition=(param,), mapping_update={param: angle}
#     )

class FourierAnsatz(ImmutableLinearMappedUnboundParametricQuantumCircuit):
    def __init__(self, hardware_type: str, n_qubits: int, layers: list[list[int]]):
        n_qubits = n_qubits
        circuit = LinearMappedUnboundParametricQuantumCircuit(n_qubits)

        for layer_idx, layer in enumerate(layers):
            if(len(layer) != len(set(layer))):
                raise RuntimeError("each layer cannot have duplicate elements")
            for i in range(n_qubits):
                if(hardware_type == "sc"):
                    # circuit.add_RY_gate(i, np.pi/2)
                    circuit.add_SqrtX_gate(i)
                else:
                    # circuit.add_gate(U1q(i, np.pi/2, np.pi/2))
                    circuit.add_gate(U1q(i, np.pi/2, 0))
            for j in range(1, len(layer), 2):
                i = j - 1
                theta_1, theta_2, theta_3 = circuit.add_parameters(
                    f"theta_{layer_idx}_{i//2}_1", f"theta_{layer_idx}_{i//2}_2", f"theta_{layer_idx}_{i//2}_3"
                )

                # diag([0, t2+t3, t1+t3, t1+t2])
                circuit.add_ParametricRZ_gate(layer[i], theta_1)
                circuit.add_ParametricRZ_gate(layer[j], theta_2)
                circuit.add_ParametricPauliRotation_gate([layer[i], layer[j]], [3, 3], theta_3)
        
        super().__init__(circuit)

def param_convert_func_FourierAnsatz(params: list[float]):
    dst = []
    for i in range(0, len(params), 3):
        j, k = i + 1, i + 2
        si, sj, sk = params[i], params[j], params[k]
        s = (si + sj + sk) / 2
        dst += [s - si, s - sj, s - sk]
    return dst