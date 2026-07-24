## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 70.53152208473419


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-20.7560692, 66.8260574, -20.7560692, 66.8260574, -87.5821228, 87.5821228)
1: (-16.0830002, 40.9191780, -16.0830002, 40.9191780, -57.0021744, 57.0021744)
2: (-16.7411327, 38.9048538, -16.7411327, 38.9048538, -55.6459846, 55.6459846)
3: (-15.2124786, 49.9498901, -15.2124786, 49.9498901, -65.1623688, 65.1623688)
4: (-25.2485352, 40.3464584, -25.2485352, 40.3464584, -65.5949936, 65.5949936)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.74 + 1.64 = 3.38 seconds
status: Status.VERIFIED
relational distance
Output dim: 3, lower bound: -60.8483692, upper bound: 60.8483692
