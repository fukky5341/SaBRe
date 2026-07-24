## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 0)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.007504829105071612


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (2.1719015, 2.2768910, 2.1719015, 2.2768910, -0.0161514, 0.0161515)
1: (-0.4074353, -0.0280401, -0.4074353, -0.0280401, -0.1123534, 0.1123534)
2: (-3.0187213, -2.7266231, -3.0187213, -2.7266231, -0.0371951, 0.0371951)
3: (-2.9631896, -2.5854878, -2.9631896, -2.5854878, -0.0845951, 0.0845951)
4: (-2.2055104, -1.7300730, -2.2055104, -1.7300730, -0.1136587, 0.1136587)
5: (-4.6386366, -4.1821027, -4.6386366, -4.1821027, -0.0908564, 0.0908564)
6: (-4.9983592, -4.3473735, -4.9983592, -4.3473735, -0.3528652, 0.3528652)
7: (-4.4868116, -3.9626141, -4.4868116, -3.9626141, -0.1161866, 0.1161866)
8: (-0.2934746, -0.0922195, -0.2934746, -0.0922195, -0.0222106, 0.0222106)
9: (-1.3636985, -1.0762317, -1.3636985, -1.0762317, -0.0458677, 0.0458677)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 9.72 + 21.34 = 31.06 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.0058848, upper bound: 0.0058870
