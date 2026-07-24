## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 4)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.43956933185382435


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.0106988, -3.0791709, -5.0106988, -3.0791709, -1.4492099, 1.4492100)
1: (-3.5415134, -0.2757564, -3.5415134, -0.2757564, -2.9340625, 2.9340627)
2: (-2.3206022, -1.2172091, -2.3206022, -1.2172091, -0.7953197, 0.7953197)
3: (-0.3319211, 0.3054654, -0.3319211, 0.3054654, -0.5330389, 0.5330390)
4: (-2.7573586, -1.4082322, -2.7573586, -1.4082322, -0.5245243, 0.5245243)
5: (-0.9705927, -0.0035533, -0.9705927, -0.0035533, -0.6698881, 0.6698881)
6: (0.2700236, 1.1110315, 0.2700236, 1.1110315, -0.4410605, 0.4410605)
7: (-2.1667745, -0.5117657, -2.1667745, -0.5117657, -1.2767136, 1.2767135)
8: (-4.9441643, -2.8037651, -4.9441643, -2.8037651, -1.2702334, 1.2702334)
9: (-1.9368241, 0.0213175, -1.9368241, 0.0213175, -1.5586140, 1.5586141)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 8.21 + 123.25 = 131.46 seconds
status: Status.VERIFIED
relational distance
Output dim: 3, lower bound: -0.3467295, upper bound: 0.3467305
