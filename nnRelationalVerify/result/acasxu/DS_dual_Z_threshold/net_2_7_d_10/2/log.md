## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.22612451055153776


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0139729, 0.1907606, -0.0139729, 0.1907606, -0.2047335, 0.2047335)
1: (-0.0410668, -0.0186802, -0.0410668, -0.0186802, -0.0223865, 0.0223865)
2: (0.0132683, 0.0330545, 0.0132683, 0.0330545, -0.0197862, 0.0197862)
3: (-0.0193515, -0.0078677, -0.0193515, -0.0078677, -0.0114837, 0.0114837)
4: (0.0127452, 0.0298558, 0.0127452, 0.0298558, -0.0171107, 0.0171107)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.49 + 0.77 = 3.26 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.1836502, upper bound: 0.1836502
