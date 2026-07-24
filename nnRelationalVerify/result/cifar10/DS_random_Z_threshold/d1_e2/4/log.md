## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 4)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.031175433027939365


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (1.3071504, 1.4709190, 1.3071504, 1.4709190, -0.0553941, 0.0553942)
1: (-2.0505912, -1.1842415, -2.0505912, -1.1842415, -0.2874890, 0.2874891)
2: (-0.3763759, -0.1029509, -0.3763759, -0.1029509, -0.0395340, 0.0395340)
3: (-1.8700377, -1.4662535, -1.8700377, -1.4662535, -0.0785571, 0.0785571)
4: (-0.6824986, -0.4061486, -0.6824986, -0.4061486, -0.0464414, 0.0464414)
5: (-2.3358288, -1.8325825, -2.3358288, -1.8325825, -0.0957247, 0.0957247)
6: (-3.5940361, -2.7849782, -3.5940361, -2.7849782, -0.0925074, 0.0925074)
7: (-1.9088181, -1.1991714, -1.9088181, -1.1991714, -0.1262317, 0.1262317)
8: (-0.3107238, 0.1258125, -0.3107238, 0.1258125, -0.3065494, 0.3065494)
9: (-2.4144392, -1.7895064, -2.4144392, -1.7895064, -0.3486998, 0.3486998)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.83 + 38.38 = 46.21 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.0241279, upper bound: 0.0241285
