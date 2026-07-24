## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 9.717269869677207


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.6005063, 1.6386203, -1.6005063, 1.6386203, -3.2391264, 3.2391264)
1: (-10.6042576, 4.1419091, -10.6042576, 4.1419091, -14.7461662, 14.7461662)
2: (-6.0699911, 3.8816235, -6.0699911, 3.8816235, -9.9516144, 9.9516144)
3: (-7.3072715, 2.9367578, -7.3072715, 2.9367578, -10.2440290, 10.2440290)
4: (-4.1786580, 3.0546839, -4.1786580, 3.0546839, -7.2333403, 7.2333412)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.22 + 1.66 = 2.88 seconds
status: Status.VERIFIED
relational distance
Output dim: 3, lower bound: -8.1800759, upper bound: 8.1800759
