## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 5359.109039479713


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3136.6748047, 2550.9006348, -3136.6748047, 2550.9006348, -5687.5751953, 5687.5751953)
1: (-2513.2941895, 2457.4460449, -2513.2941895, 2457.4460449, -4970.7402344, 4970.7402344)
2: (-3595.4755859, 2672.1398926, -3595.4755859, 2672.1398926, -6267.6152344, 6267.6152344)
3: (-1407.5072021, 3564.2558594, -1407.5072021, 3564.2558594, -4971.7631836, 4971.7631836)
4: (-3999.4067383, 2627.0041504, -3999.4067383, 2627.0041504, -6626.4111328, 6626.4111328)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.79 + 2.18 = 2.98 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -4552.2782408, upper bound: 4552.2782408
