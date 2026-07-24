## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 19.448094130593915


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.5708892, 13.9244061, -3.5708892, 13.9244061, -17.4952946, 17.4952946)
1: (-10.5180368, 28.8611164, -10.5180368, 28.8611164, -39.3791542, 39.3791542)
2: (-16.2917290, 27.0998936, -16.2917290, 27.0998936, -43.3916245, 43.3916245)
3: (-8.7987671, 33.7294617, -8.7987671, 33.7294617, -42.5282211, 42.5282211)
4: (-14.6446037, 23.3109703, -14.6446037, 23.3109703, -37.9555740, 37.9555740)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.92 + 1.58 = 2.50 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -15.3926192, upper bound: 15.3926192
