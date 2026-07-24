## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.0476680485653473


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.8123617, -6.2212009, -8.8123617, -6.2212009, -2.1485600, 2.1485600)
1: (-12.5776806, -9.6193066, -12.5776806, -9.6193066, -2.2878542, 2.2878542)
2: (-7.5704379, -5.1113300, -7.5704379, -5.1113300, -2.4591079, 2.4591079)
3: (-9.1119881, -5.9411378, -9.1119881, -5.9411378, -2.8684578, 2.8684573)
4: (-10.3934450, -7.7036209, -10.3934450, -7.7036209, -2.5133648, 2.5133648)
5: (0.5242939, 2.7595601, 0.5242939, 2.7595601, -2.1076455, 2.1076455)
6: (5.1939564, 7.2063017, 5.1939564, 7.2063017, -1.8181987, 1.8181992)
7: (-17.5598259, -15.0159283, -17.5598259, -15.0159283, -2.1776104, 2.1776099)
8: (0.8089163, 3.4791784, 0.8089163, 3.4791784, -2.3976979, 2.3976970)
9: (-8.3383198, -6.2303343, -8.3383198, -6.2303343, -1.8262186, 1.8262186)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.92 + 33.59 = 55.51 seconds
status: Status.VERIFIED
relational distance
Output dim: 6, lower bound: -0.9178607, upper bound: 0.9178598
