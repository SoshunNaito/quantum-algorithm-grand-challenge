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

from BaseAnsatz import BaseAnsatz

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

def param_convert_func_FourierAnsatz(params: list[float]):
    dst = []
    for i in range(0, len(params), 3):
        j, k = i + 1, i + 2
        si, sj, sk = params[i], params[j], params[k]
        s = (si + sj + sk) / 2
        dst += [s - si, s - sj, s - sk]
    return dst

class FourierAnsatz(ImmutableLinearMappedUnboundParametricQuantumCircuit, BaseAnsatz):
    # def approx_cnot(self, hardware_type: str, circuit: LinearMappedUnboundParametricQuantumCircuit, a: int, b: int):
    #     if(hardware_type == "sc"):
    #         circuit.add_CNOT_gate(a, b)
    #     else:
    #         circuit.add_gate(U1q(a, np.pi/2, -np.pi/2))
    #         circuit.add_gate(ZZ(a, b))
    #         circuit.add_gate(U1q(a, np.pi/2, np.pi))

    # def multiplexed_rotation(
    #     self, hardware_type: str, circuit: LinearMappedUnboundParametricQuantumCircuit,
    #     a: int, b:int, c: int, d: int
    # ):
    #     thetas = np.random.random(15)
    #     circuit.add_gate(RZZ(a, d, thetas[0]))
    #     circuit.add_gate(RZZ(b, c, thetas[1]))
    #     circuit.add_RZ_gate(a, thetas[2])
    #     circuit.add_RZ_gate(c, thetas[3])
    #     self.approx_cnot(hardware_type, circuit, a, b)
    #     self.approx_cnot(hardware_type, circuit, c, d)
    #     circuit.add_gate(RZZ(a, c, thetas[4]))

    #     circuit.add_gate(RZZ(a, d, thetas[5]))
    #     circuit.add_gate(RZZ(b, c, thetas[6]))
    #     circuit.add_RZ_gate(b, thetas[7])
    #     circuit.add_RZ_gate(d, thetas[8])
    #     self.approx_cnot(hardware_type, circuit, b, a)
    #     self.approx_cnot(hardware_type, circuit, d, c)
    #     circuit.add_gate(RZZ(b, d, thetas[9]))

    #     circuit.add_gate(RZZ(a, d, thetas[10]))
    #     circuit.add_gate(RZZ(b, c, thetas[11]))
    #     circuit.add_RZ_gate(a, thetas[12])
    #     circuit.add_RZ_gate(c, thetas[13])
    #     self.approx_cnot(hardware_type, circuit, a, b)
    #     self.approx_cnot(hardware_type, circuit, c, d)
    #     circuit.add_gate(RZZ(a, c, thetas[14]))

    def __init__(
        self, hardware_type: str, n_qubits: int, qubit_to_coord: dict[int, tuple[int, int]],
        layers_before: list[list[int]], swap_layer: list[int], layers_after: list[list[int]]
    ):
        n_qubits = n_qubits
        circuit = LinearMappedUnboundParametricQuantumCircuit(n_qubits)
        self.qubit_to_coord = qubit_to_coord
        self.parameter_convert_func = lambda params: param_convert_func_FourierAnsatz(params)
        self.num_single_qubit_gates = 0
        self.num_multi_qubit_gates = 0
        self.num_parameters = 0

        # if(n_qubits == 4):
        #     self.multiplexed_rotation(hardware_type, circuit, 0, 1, 2, 3)
        # else:
        #     for i in range(n_qubits):
        #         if(hardware_type == "sc"):
        #             # circuit.add_RY_gate(i, np.pi/2)
        #             circuit.add_SqrtX_gate(i)
        #         else:
        #             # circuit.add_gate(U1q(i, np.pi/2, np.pi/2))
        #             circuit.add_gate(U1q(i, np.pi/2, 0))
        #         self.num_single_qubit_gates += 1
        #     self.multiplexed_rotation(hardware_type, circuit, 0, 1, 4, 5)
        #     self.multiplexed_rotation(hardware_type, circuit, 2, 3, 6, 7)

        for layer_idx, layer in enumerate(layers_before + layers_after):
            if(len(layer) != n_qubits):
                raise RuntimeError("invalid layer size:", layer)
            if(len(layer) != len(set(layer))):
                raise RuntimeError("each layer cannot have duplicate elements")
            
            for i in range(n_qubits):
                if(hardware_type == "sc"):
                    # circuit.add_RY_gate(i, np.pi/2)
                    circuit.add_SqrtX_gate(i)
                else:
                    # circuit.add_gate(U1q(i, np.pi/2, np.pi/2))
                    circuit.add_gate(U1q(i, np.pi/2, 0))
                self.num_single_qubit_gates += 1

            for j in range(1, len(layer), 2):
                i = j - 1
                theta_1, theta_2, theta_3 = circuit.add_parameters(
                    f"theta_{layer_idx}_{i//2}_1", f"theta_{layer_idx}_{i//2}_2", f"theta_{layer_idx}_{i//2}_3"
                )
                self.num_parameters += 3

                # diag([0, t2+t3, t1+t3, t1+t2])
                qi, qj = layer[i], layer[j]
                if(hardware_type == "sc" and layer_idx < len(layers_before)):
                    if qi in swap_layer:
                        idx0 = swap_layer.index(qi)
                        idx1 = idx0 + 1 if idx0 % 2 == 0 else idx0 - 1
                        qi = swap_layer[idx1]
                    if qj in swap_layer:
                        idx0 = swap_layer.index(qj)
                        idx1 = idx0 + 1 if idx0 % 2 == 0 else idx0 - 1
                        qj = swap_layer[idx1]
                circuit.add_ParametricRZ_gate(qi, theta_1)
                circuit.add_ParametricRZ_gate(qj, theta_2)
                circuit.add_ParametricPauliRotation_gate([qi, qj], [3, 3], theta_3)

                self.num_single_qubit_gates += 3 if hardware_type == "sc" else 2
                self.num_multi_qubit_gates += 2 if hardware_type == "sc" else 1
            
            if(hardware_type == "sc" and layer_idx == len(layers_before) - 1):
                for j in range(1, len(swap_layer), 2):
                    i = j - 1
                    circuit.add_SWAP_gate(swap_layer[i], swap_layer[j])
                    self.num_multi_qubit_gates += 3
        
        # if(n_qubits == 8):
        #     # for i in range(n_qubits):
        #     #     if(hardware_type == "sc"):
        #     #         # circuit.add_RY_gate(i, np.pi/2)
        #     #         circuit.add_SqrtX_gate(i)
        #     #     else:
        #     #         # circuit.add_gate(U1q(i, np.pi/2, np.pi/2))
        #     #         circuit.add_gate(U1q(i, np.pi/2, 0))
        #     #     self.num_single_qubit_gates += 1
        #     # self.multiplexed_rotation(hardware_type, circuit, 0, 1, 4, 5)
        #     # self.multiplexed_rotation(hardware_type, circuit, 2, 3, 6, 7)
        #     for i in range(n_qubits):
        #         if(hardware_type == "sc"):
        #             # circuit.add_RY_gate(i, np.pi/2)
        #             circuit.add_SqrtX_gate(i)
        #         else:
        #             # circuit.add_gate(U1q(i, np.pi/2, np.pi/2))
        #             circuit.add_gate(U1q(i, np.pi/2, 0))
        #         self.num_single_qubit_gates += 1
        #     self.multiplexed_rotation(hardware_type, circuit, 0, 3, 4, 7)
        #     self.multiplexed_rotation(hardware_type, circuit, 1, 2, 5, 6)
        
        super().__init__(circuit)

class FourierAnsatz_4(FourierAnsatz):
    def __init__(self, hardware_type: str):
        #   0 3   #
        #   1 2   #
        #    ↓    #
        #   0 2   #
        #   1 3   #
        qubit_to_coord = {0: (0,0), 1: (1,0), 2: (0,1), 3: (1,1)}
        layers_before = [[0, 1, 2, 3], [0, 3, 1, 2]]
        swap_layer = [2, 3]
        layers_after = [[0, 2, 1, 3], [0, 1, 2, 3]]
        
        super().__init__(
            hardware_type, 4,
            qubit_to_coord,
            layers_before, swap_layer, layers_after
        )

class FourierAnsatz_8(FourierAnsatz):
    def __init__(self, hardware_type: str):
        #   0 4 2 6   #
        #   1 5 3 7   #
        #      ↓      #
        #   0 2 4 6   #
        #   1 3 5 7   #

        #   0 2 1 3   #
        #   6 4 7 5   #
        #      ↓      #
        #   0 1 2 3   #
        #   6 7 4 5   #

        # -7.7くらい
        # #     1   3   #
        # #   0   2     #
        # #     7   5   #
        # #   6   4     #
        # qubit_to_coord = {0: (0,0), 1: (0,1), 2: (0,2), 3: (0,3), 4: (1,2), 5: (1,3), 6: (1,0), 7: (1,1)}
        # layers_before = [[0, 2, 1, 3, 4, 6, 5, 7], [0, 6, 1, 7, 2, 4, 3, 5], [0, 1, 2, 3, 4, 5, 6, 7]]
        # swap_layer = [1, 2, 4, 7]
        # layers_after = [[0, 2, 1, 3, 4, 6, 5, 7], [0, 6, 1, 7, 2, 4, 3, 5], [0, 1, 2, 3, 4, 5, 6, 7]]

        # #     1   2   #
        # #   0   3     #
        # #     5   6   #
        # #   4   7     #
        # qubit_to_coord = {0: (0,0), 1: (0,1), 2: (0,2), 3: (0,3), 4: (1,2), 5: (1,3), 6: (1,0), 7: (1,1)}
        # layers_before = [[0, 4, 1, 5, 2, 6, 3, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 3, 1, 2, 4, 7, 5, 6]]
        # swap_layer = [1, 2, 4, 7]
        # layers_after = [[0, 4, 1, 5, 2, 6, 3, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 3, 1, 2, 4, 7, 5, 6]]

        #     1   3   #
        #   0   2     #
        #     5   7   #
        #   4   6     #
        qubit_to_coord = {0: (0,0), 1: (0,1), 2: (0,2), 3: (0,3), 4: (1,2), 5: (1,3), 6: (1,0), 7: (1,1)}
        layers_before = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 6, 1, 2, 3, 5, 4, 7], [0, 2, 1, 3, 4, 6, 5, 7], [0, 6, 1, 4, 3, 5, 2, 7], [0, 4, 1, 5, 2, 6, 3, 7]]
        swap_layer = [1, 2, 4, 7]
        layers_after = [[0, 7, 1, 6, 2, 5, 3, 4], [0, 5, 1, 7, 3, 6, 2, 4], [0, 3, 1, 2, 4, 7, 5, 6]]
        
        super().__init__(
            hardware_type, 8,
            qubit_to_coord,
            layers_before, swap_layer, layers_after
        )