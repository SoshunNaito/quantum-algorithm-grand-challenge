from itertools import permutations
from typing import Tuple

positions_ans = set()
positions_all = set([str(list(x)) for x in permutations([0, 0, 0, 0, 1, 1, 1, 1])])

gates: list[Tuple[int, int]] = [
    (0, 4), (1, 5), (2, 6), (3, 7),
    (0, 1), (2, 3), (4, 6), (5, 7),
]
init = [0, 0, 0, 0, 1, 1, 1, 1]

for i in range(2**len(gates)):
    x: list[int] = init.copy()
    for j in range(len(gates)):
        a, b = gates[j]
        if(i & (1 << j)):
            x[a], x[b] = x[b], x[a]
    positions_ans.add(str(x))

print(len(positions_ans))
print(positions_ans)
# positions_diff = positions_all - positions_ans
# print(positions_diff)