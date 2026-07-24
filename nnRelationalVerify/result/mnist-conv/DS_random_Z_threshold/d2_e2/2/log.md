## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.19283703708165037


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.3653331, -6.6729102, -7.3653331, -6.6729102, -0.3814726, 0.3814726)
1: (3.3846326, 3.8152595, 3.3846326, 3.8152595, -0.2590060, 0.2590060)
2: (-6.3814130, -5.8416061, -6.3814130, -5.8416061, -0.3504484, 0.3504484)
3: (-9.6910591, -8.9502630, -9.6910591, -8.9502630, -0.3364486, 0.3364486)
4: (-4.1883917, -3.6298001, -4.1883917, -3.6298001, -0.2581156, 0.2581156)
5: (-10.1242027, -9.3844090, -10.1242027, -9.3844090, -0.2859797, 0.2859797)
6: (-7.3823271, -6.7658138, -7.3823271, -6.7658138, -0.2511786, 0.2511786)
7: (-6.1901946, -5.7462568, -6.1901946, -5.7462568, -0.3057592, 0.3057592)
8: (-0.7283597, -0.1443071, -0.7283597, -0.1443071, -0.2805092, 0.2805092)
9: (-7.0876017, -6.4874830, -7.0876017, -6.4874830, -0.3066001, 0.3066001)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.75 + 32.38 = 57.13 seconds
status: Status.VERIFIED
relational distance
Output dim: 1, lower bound: -0.1520236, upper bound: 0.1520231
