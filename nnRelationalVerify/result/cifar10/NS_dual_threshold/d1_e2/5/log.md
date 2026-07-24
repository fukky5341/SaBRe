## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 5)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.009572678279870569


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.1846647, -3.4223347, -4.1846647, -3.4223347, -0.2079982, 0.2079982)
1: (-3.2684379, -2.0738583, -3.2684379, -2.0738583, -0.5437425, 0.5437425)
2: (-1.8146329, -1.4243026, -1.8146329, -1.4243026, -0.0746192, 0.0746192)
3: (-0.1459303, 0.2119474, -0.1459303, 0.2119474, -0.2483499, 0.2483499)
4: (-1.7856511, -1.5382342, -1.7856511, -1.5382342, -0.0687176, 0.0687176)
5: (-1.0305028, -0.5649048, -1.0305028, -0.5649048, -0.3328762, 0.3328762)
6: (0.9667889, 1.0866712, 0.9667889, 1.0866712, -0.0180829, 0.0180829)
7: (-1.7863696, -1.1963891, -1.7863696, -1.1963891, -0.1593086, 0.1593085)
8: (-3.4036913, -2.4049263, -3.4036913, -2.4049263, -0.3764695, 0.3764695)
9: (-0.7621465, 0.1715422, -0.7621465, 0.1715422, -0.5216683, 0.5216683)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.11 + 41.07 = 48.19 seconds
status: Status.VERIFIED
relational distance
Output dim: 6, lower bound: -0.0079275, upper bound: 0.0079291
