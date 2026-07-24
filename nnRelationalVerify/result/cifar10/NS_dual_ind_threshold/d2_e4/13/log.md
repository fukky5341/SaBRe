## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 13)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.11840017993880658


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.0521257, -0.9090614, -3.0521257, -0.9090614, -1.1440459, 1.1440461)
1: (-4.9254699, -1.6117833, -4.9254699, -1.6117833, -2.0082960, 2.0082960)
2: (-2.2591529, -1.3810029, -2.2591529, -1.3810029, -0.6092996, 0.6092995)
3: (-3.5013642, -2.1288342, -3.5013642, -2.1288342, -1.0182114, 1.0182114)
4: (-1.2644778, -0.6234192, -1.2644778, -0.6234192, -0.4211028, 0.4211028)
5: (-3.1448703, -1.3762714, -3.1448703, -1.3762714, -1.3117647, 1.3117647)
6: (-7.6828923, -5.2051959, -7.6828923, -5.2051959, -1.1495214, 1.1495212)
7: (1.1253493, 2.7666342, 1.1253493, 2.7666342, -0.9491487, 0.9491487)
8: (-3.3694675, -0.2727237, -3.3694675, -0.2727237, -2.3432992, 2.3432992)
9: (-1.2787340, 0.3824947, -1.2787340, 0.3824947, -1.3546004, 1.3546002)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 9.67 + 41.89 = 51.56 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.1027721, upper bound: 0.1027735
