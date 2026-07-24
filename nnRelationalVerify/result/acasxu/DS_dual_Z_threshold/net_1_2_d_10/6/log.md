## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 6188.337627129417


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-863.1370239, 3753.3532715, -863.1370239, 3753.3532715, -4616.4902344, 4616.4902344)
1: (-1086.8886719, 4259.3261719, -1086.8886719, 4259.3261719, -5346.2148438, 5346.2148438)
2: (-1105.0872803, 4253.2548828, -1105.0872803, 4253.2548828, -5358.3417969, 5358.3417969)
3: (-1736.3594971, 4426.5043945, -1736.3594971, 4426.5043945, -6162.8627930, 6162.8627930)
4: (-1741.8470459, 4238.6694336, -1741.8470459, 4238.6694336, -5980.5166016, 5980.5166016)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.78 + 2.42 = 3.20 seconds
status: Status.VERIFIED
relational distance
Output dim: 3, lower bound: -5471.9784739, upper bound: 5471.9784739
