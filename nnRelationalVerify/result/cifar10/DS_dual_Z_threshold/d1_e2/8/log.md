## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 8)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.006091288679876992


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.0954549, -2.5307403, -3.0954549, -2.5307403, -0.2720383, 0.2720384)
1: (-3.9853008, -3.0967598, -3.9853008, -3.0967598, -0.3919992, 0.3919992)
2: (0.0278484, 0.2212955, 0.0278484, 0.2212955, -0.0979383, 0.0979383)
3: (-1.2939787, -1.0578853, -1.2939787, -1.0578853, -0.1018504, 0.1018504)
4: (0.1459976, 0.3057243, 0.1459976, 0.3057243, -0.0402210, 0.0402210)
5: (-1.4264108, -1.2061805, -1.4264108, -1.2061805, -0.1030801, 0.1030801)
6: (0.3661533, 0.4477126, 0.3661533, 0.4477126, -0.0176174, 0.0176174)
7: (-1.5409867, -1.2077650, -1.5409867, -1.2077650, -0.0585659, 0.0585659)
8: (-4.8373985, -4.3697524, -4.8373985, -4.3697524, -0.2195679, 0.2195679)
9: (-3.8952889, -3.4271805, -3.8952889, -3.4271805, -0.2216583, 0.2216583)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.20 + 18.70 = 25.90 seconds
status: Status.VERIFIED
relational distance
Output dim: 6, lower bound: -0.0051400, upper bound: 0.0051415
