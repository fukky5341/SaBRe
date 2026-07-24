## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.9588488454187355


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.6862783, -4.5913610, -6.6862783, -4.5913610, -1.6763077, 1.6763082)
1: (-13.1455326, -10.3723631, -13.1455326, -10.3723631, -2.1569643, 2.1569643)
2: (-6.7355099, -4.5157666, -6.7355099, -4.5157666, -1.6031456, 1.6031456)
3: (-12.3076057, -10.2357445, -12.3076057, -10.2357445, -1.4699936, 1.4699941)
4: (-6.4482555, -4.1263995, -6.4482555, -4.1263995, -1.5913258, 1.5913253)
5: (-2.3221331, -0.4494164, -2.3221331, -0.4494164, -1.3761282, 1.3761282)
6: (9.5122805, 11.7396517, 9.5122805, 11.7396517, -1.7478533, 1.7478542)
7: (-18.0711174, -15.7608347, -18.0711174, -15.7608347, -1.6263638, 1.6263638)
8: (-0.8006053, 1.0504651, -0.8006053, 1.0504651, -1.4443054, 1.4443054)
9: (-15.5792732, -13.1956453, -15.5792732, -13.1956453, -1.7956176, 1.7956171)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.55 + 45.24 = 68.79 seconds
status: Status.VERIFIED
relational distance
Output dim: 6, lower bound: -0.7764339, upper bound: 0.7764351
