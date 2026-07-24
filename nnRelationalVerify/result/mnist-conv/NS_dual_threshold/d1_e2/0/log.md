## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.14876789841729493


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.0047760, -7.6571732, -8.0047760, -7.6571732, -0.1932001, 0.1932001)
1: (2.7412825, 3.1104660, 2.7412825, 3.1104660, -0.1697667, 0.1697669)
2: (-7.1127520, -6.7995734, -7.1127520, -6.7995734, -0.1666405, 0.1666405)
3: (-11.5315361, -11.1812077, -11.5315361, -11.1812077, -0.1523087, 0.1523087)
4: (-4.9379959, -4.5408764, -4.9379959, -4.5408764, -0.2248106, 0.2248106)
5: (-8.8896933, -8.5877743, -8.8896933, -8.5877743, -0.1095153, 0.1095153)
6: (-5.9075637, -5.5288715, -5.9075637, -5.5288715, -0.1735747, 0.1735746)
7: (-4.8020935, -4.5400219, -4.8020935, -4.5400219, -0.1040741, 0.1040742)
8: (-2.3820438, -2.0826368, -2.3820438, -2.0826368, -0.1289288, 0.1289289)
9: (-10.6855688, -10.2598848, -10.6855688, -10.2598848, -0.2645135, 0.2645135)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.99 + 33.07 = 55.05 seconds
status: Status.VERIFIED
relational distance
Output dim: 1, lower bound: -0.1178287, upper bound: 0.1178285
