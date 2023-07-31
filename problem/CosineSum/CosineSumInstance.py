import math

class CosineSumInstance:
    def __init__(
        self,
        quad_amplitudes: list[list[float]], quad_offsets: list[list[float]],
        linear_amplitudes: list[float], linear_offsets: list[float],
        constant: float
    ):
        self.num_theta_params = len(quad_amplitudes)
        assert len(quad_offsets) == self.num_theta_params
        assert len(linear_amplitudes) == self.num_theta_params
        assert len(linear_offsets) == self.num_theta_params
        for i in range(self.num_theta_params):
            assert len(quad_amplitudes[i]) == self.num_theta_params
            assert len(quad_offsets[i]) == self.num_theta_params
        
        self.quad_amplitudes = quad_amplitudes
        self.quad_offsets = quad_offsets
        self.linear_amplitudes = linear_amplitudes
        self.linear_offsets = linear_offsets
        self.constant = constant

    def eval(self, theta_params: list[float]) -> float:
        assert len(theta_params) == self.num_theta_params
        ans = 0
        for i in range(self.num_theta_params):
            for j in range(i + 1, self.num_theta_params):
                ans += self.quad_amplitudes[i][j] * math.cos(theta_params[i] - theta_params[j] + self.quad_offsets[i][j])
            ans += self.linear_amplitudes[i] * math.cos(theta_params[i] + self.linear_offsets[i])
        ans += self.constant        
        return ans

    def grad(self, theta_params: list[float]) -> list[float]:
        assert len(theta_params) == self.num_theta_params
        ans = [0.0] * self.num_theta_params
        for i in range(self.num_theta_params):
            for j in range(i + 1, self.num_theta_params):
                tmp = self.quad_amplitudes[i][j] * math.sin(theta_params[i] - theta_params[j] + self.quad_offsets[i][j])
                ans[i] -= tmp
                ans[j] += tmp
            ans[i] -= self.linear_amplitudes[i] * math.sin(theta_params[i] + self.linear_offsets[i])
        return ans
    
    def hess(self, theta_params: list[float]) -> list[list[float]]:
        assert len(theta_params) == self.num_theta_params
        ans = [[0.0 for _ in range(self.num_theta_params)] for _ in range(self.num_theta_params)]
        for i in range(self.num_theta_params):
            for j in range(i + 1, self.num_theta_params):
                tmp = self.quad_amplitudes[i][j] * math.cos(theta_params[i] - theta_params[j] + self.quad_offsets[i][j])
                ans[i][j] = tmp
                ans[j][i] = tmp
                ans[i][i] -= tmp
                ans[j][j] -= tmp
            ans[i][i] -= self.linear_amplitudes[i] * math.cos(theta_params[i] + self.linear_offsets[i])
        return ans