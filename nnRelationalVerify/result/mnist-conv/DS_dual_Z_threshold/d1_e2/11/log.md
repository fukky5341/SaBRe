## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.20775074026449103


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-11.1331463, -10.5543909, -11.1331463, -10.5543909, -0.3351262, 0.3351262)
1: (-9.9766531, -9.5031185, -9.9766531, -9.5031185, -0.3055673, 0.3055673)
2: (-9.5403452, -9.0933914, -9.5403452, -9.0933914, -0.3239734, 0.3239734)
3: (-8.9054670, -8.4085274, -8.9054670, -8.4085274, -0.3439398, 0.3439400)
4: (-1.9130120, -1.4867046, -1.9130120, -1.4867046, -0.3613234, 0.3613234)
5: (-5.7405481, -5.3692274, -5.7405481, -5.3692274, -0.2740374, 0.2740374)
6: (-10.8034897, -10.2895918, -10.8034897, -10.2895918, -0.2804530, 0.2804530)
7: (-4.9775810, -4.4842062, -4.9775810, -4.4842062, -0.3468432, 0.3468432)
8: (-0.9739132, -0.6251135, -0.9739132, -0.6251135, -0.2362905, 0.2362905)
9: (4.0503368, 4.4993114, 4.0503368, 4.4993114, -0.3724418, 0.3724418)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.16 + 33.41 = 55.57 seconds
status: Status.VERIFIED
relational distance
Output dim: 9, lower bound: -0.1780588, upper bound: 0.1780591
