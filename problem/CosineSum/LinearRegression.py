import numpy as np


def LinearRegression(
    num_variables: int,
    num_samples: int,
    input_list: list[list[float]],
    output_list: list[float],
) -> list[float]:
    assert len(input_list) == num_samples and len(output_list) == num_samples
    assert len(input_list[0]) == num_variables
    X = np.array(input_list)
    Y = np.array(output_list)
    A = X.T @ X

    U, S, V_t = np.linalg.svd(A, full_matrices=False)

    while S[-1] < 0.00000001: S = S[:-1]
    r = len(S)
    # print("rank = " + str(r))

    U = U[:, :r]
    S_inv = np.diag([1.0 / s for s in S])
    V_t = V_t[:r, :]

    A_inv = V_t.T @ S_inv @ U.T

    ans = A_inv @ X.T @ Y
    return list(ans)
