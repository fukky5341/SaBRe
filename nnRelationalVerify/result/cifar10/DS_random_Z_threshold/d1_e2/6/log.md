## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 6)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.013670709250712737


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.6014524, -1.3128889, -1.6014524, -1.3128889, -0.0391597, 0.0391597)
1: (-0.0523789, 0.1202041, -0.0523789, 0.1202041, -0.0295876, 0.0295876)
2: (-2.4075322, -2.0202446, -2.4075322, -2.0202446, -0.0521877, 0.0521877)
3: (-1.7881825, -1.3572050, -1.7881825, -1.3572050, -0.0861368, 0.0861368)
4: (-5.3692827, -4.7256074, -5.3692827, -4.7256074, -0.0760576, 0.0760576)
5: (-1.7423579, -1.2880762, -1.7423579, -1.2880762, -0.0908374, 0.0908374)
6: (-2.4585419, -1.9506505, -2.4585419, -1.9506505, -0.2490501, 0.2490501)
7: (-2.8043444, -2.2621493, -2.8043444, -2.2621493, -0.0914789, 0.0914789)
8: (0.0922084, 0.2942025, 0.0922084, 0.2942025, -0.0205747, 0.0205747)
9: (0.5746570, 0.8365768, 0.5746570, 0.8365768, -0.1687023, 0.1687023)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.77 + 17.80 = 25.57 seconds
status: Status.VERIFIED
relational distance
Output dim: 8, lower bound: -0.0106447, upper bound: 0.0106452
