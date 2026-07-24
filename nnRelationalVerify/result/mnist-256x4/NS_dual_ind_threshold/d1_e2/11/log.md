## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.008336198261795325


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0048467, -0.0011076, -0.0048467, -0.0011076, -0.0037391, 0.0037391)
1: (0.0008642, 0.0086216, 0.0008642, 0.0086216, -0.0077575, 0.0077575)
2: (0.0075051, 0.0157675, 0.0075051, 0.0157675, -0.0068037, 0.0068037)
3: (-0.0057674, -0.0030647, -0.0057674, -0.0030647, -0.0027027, 0.0027027)
4: (0.0032912, 0.0061013, 0.0032912, 0.0061013, -0.0025140, 0.0025140)
5: (-0.0026993, 0.0011465, -0.0026993, 0.0011465, -0.0038458, 0.0038458)
6: (-0.0062755, -0.0044620, -0.0062755, -0.0044620, -0.0018135, 0.0018135)
7: (-0.0038286, 0.0003811, -0.0038286, 0.0003811, -0.0042097, 0.0042097)
8: (-0.0048663, -0.0010354, -0.0048663, -0.0010354, -0.0038309, 0.0038309)
9: (0.9961787, 1.0063196, 0.9961787, 1.0063196, -0.0098525, 0.0098525)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.25 + 2.04 = 3.28 seconds
status: Status.VERIFIED
relational distance
Output dim: 9, lower bound: -0.0072036, upper bound: 0.0072036
