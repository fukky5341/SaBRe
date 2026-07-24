## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.38670839677831165


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.4055357, -6.2547216, -7.4055357, -6.2547216, -0.7107708, 0.7107708)
1: (-5.3230915, -4.2918730, -5.3230915, -4.2918730, -0.7402022, 0.7402020)
2: (-6.6889849, -5.2708244, -6.6889849, -5.2708244, -0.8624325, 0.8624325)
3: (-12.1245022, -10.9565516, -12.1245022, -10.9565516, -0.8732507, 0.8732505)
4: (7.4603071, 8.2828922, 7.4603071, 8.2828922, -0.5402064, 0.5402064)
5: (-7.9910469, -6.8077974, -7.9910469, -6.8077974, -0.7366507, 0.7366507)
6: (-6.0007982, -4.6604347, -6.0007982, -4.6604347, -0.8147907, 0.8147910)
7: (-6.8755517, -5.7360163, -6.8755517, -5.7360163, -0.7608571, 0.7608571)
8: (-6.1072817, -4.7903848, -6.1072817, -4.7903848, -1.0025740, 1.0025740)
9: (-6.1263270, -5.2401013, -6.1263270, -5.2401013, -0.5114366, 0.5114365)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.30 + 33.24 = 56.54 seconds
status: Status.VERIFIED
relational distance
Output dim: 4, lower bound: -0.3423241, upper bound: 0.3423245
