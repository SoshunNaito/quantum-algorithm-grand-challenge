import sys
sys.path.append("../")
import numpy as np
import math

from CosineSum.CosineSumInstance import CosineSumInstance
from CosineSum.LinearRegression import LinearRegression

from typing import Callable


def GenerateCosineSumInstance(
    num_theta_params: int,
    has_linear_term: bool,
    measurement_func: Callable[[list[float]], float],
    num_additional_measurement: int = 0
) -> CosineSumInstance:
    num_variables = num_theta_params * (num_theta_params + (1 if has_linear_term else -1)) + 1
    num_samples = num_variables + num_additional_measurement

    input_list, output_list = [], []
    for sample_idx in range(num_samples):
        _input = []
        input_params = list(np.random.rand(num_theta_params) * np.pi * 2)
        for i in range(len(input_params)):
            for j in range(i + 1, len(input_params)):
                _input.append(math.cos(input_params[i] - input_params[j]))
                _input.append(math.sin(input_params[i] - input_params[j]))
        if(has_linear_term):
            for i in range(len(input_params)):
                _input.append(math.cos(input_params[i]))
                _input.append(math.sin(input_params[i]))
        _input.append(1)

        input_list.append(_input)
        output_list.append(measurement_func(input_params))

    array_1d = LinearRegression(num_variables, num_samples, input_list, output_list)
    quad_coefs = array_1d[:num_theta_params * (num_theta_params - 1)]
    linear_coefs = array_1d[num_theta_params * (num_theta_params - 1) : -1] if has_linear_term else [0] * (num_theta_params * 2)
    constant = array_1d[-1]

    quad_amplitudes: list[list[float]] = [[0] * num_theta_params for _ in range(num_theta_params)]
    quad_offsets: list[list[float]] = [[0] * num_theta_params for _ in range(num_theta_params)]
    linear_amplitudes: list[float] = [0] * num_theta_params
    linear_offsets: list[float] = [0] * num_theta_params

    counter = 0
    for i in range(num_theta_params):
        for j in range(i + 1, num_theta_params):
            x, y = quad_coefs[counter * 2], quad_coefs[counter * 2 + 1]
            quad_amplitudes[i][j] = math.sqrt(x ** 2 + y ** 2)
            quad_amplitudes[j][i] = quad_amplitudes[i][j]
            quad_offsets[i][j] = math.atan2(-y, x)
            quad_offsets[j][i] = -quad_offsets[i][j]
            counter += 1
    if(has_linear_term):
        for i in range(num_theta_params):
            x, y = linear_coefs[i * 2], linear_coefs[i * 2 + 1]
            linear_amplitudes[i] = math.sqrt(x ** 2 + y ** 2)
            linear_offsets[i] = math.atan2(-y, x)
    
    return CosineSumInstance(
        quad_amplitudes, quad_offsets,
        linear_amplitudes, linear_offsets,
        constant
    )
