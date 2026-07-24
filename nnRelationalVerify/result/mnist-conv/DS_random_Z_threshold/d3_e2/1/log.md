## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.579440424562645


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.7544675, -3.5090306, -4.7544675, -3.5090306, -1.0169597, 1.0169597)
1: (-5.6167011, -4.5270281, -5.6167011, -4.5270281, -0.7890949, 0.7890948)
2: (-1.5062153, -0.1842887, -1.5062153, -0.1842887, -0.8368101, 0.8368099)
3: (-2.6061141, -0.9702461, -2.6061141, -0.9702461, -1.2012053, 1.2012053)
4: (-11.0932379, -9.3703299, -11.0932379, -9.3703299, -1.1436787, 1.1436787)
5: (-7.0589752, -5.7657375, -7.0589752, -5.7657375, -0.8238084, 0.8238084)
6: (-5.7908783, -4.3514996, -5.7908783, -4.3514996, -1.0459661, 1.0459659)
7: (-7.0073867, -5.6193562, -7.0073867, -5.6193562, -0.9298518, 0.9298515)
8: (7.1309848, 8.2745028, 7.1309848, 8.2745028, -0.7474172, 0.7474172)
9: (-10.9440241, -9.4617119, -10.9440241, -9.4617119, -1.1292400, 1.1292400)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.79 + 34.45 = 58.24 seconds
status: Status.VERIFIED
relational distance
Output dim: 8, lower bound: -0.4629664, upper bound: 0.4629666
