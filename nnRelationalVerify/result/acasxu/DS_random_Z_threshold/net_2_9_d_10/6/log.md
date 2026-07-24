## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 5065.9748895346


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-25111.1152344, 19474.1562500, -25111.1152344, 19474.1562500, -44585.2695312, 44585.2695312)
1: (-2098.6848145, 1499.1213379, -2098.6848145, 1499.1213379, -3597.8061523, 3597.8061523)
2: (-1384.1658936, 2482.2626953, -1384.1658936, 2482.2626953, -3866.4284668, 3866.4284668)
3: (-1663.6562500, 3654.5661621, -1663.6562500, 3654.5661621, -5318.2216797, 5318.2211914)
4: (-1365.6164551, 2572.1169434, -1365.6164551, 2572.1169434, -3937.7331543, 3937.7331543)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.11 + 2.28 = 3.39 seconds
status: Status.VERIFIED
relational distance
Output dim: 3, lower bound: -4479.3213161, upper bound: 4479.3213156
