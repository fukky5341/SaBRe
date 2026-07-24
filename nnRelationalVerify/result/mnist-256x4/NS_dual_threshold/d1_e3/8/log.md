## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00135751


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0046414, 0.0067363, 0.0046414, 0.0067363, -0.0016412, 0.0016412)
1: (0.0019929, 0.0022955, 0.0019929, 0.0022955, -0.0002371, 0.0002371)
2: (0.0106356, 0.0117937, 0.0106356, 0.0117937, -0.0009074, 0.0009074)
3: (-0.0036807, -0.0024828, -0.0036807, -0.0024828, -0.0009384, 0.0009384)
4: (-0.0013492, -0.0000524, -0.0013492, -0.0000524, -0.0010159, 0.0010159)
5: (0.0041630, 0.0053901, 0.0041630, 0.0053901, -0.0009614, 0.0009614)
6: (-0.0057829, -0.0009139, -0.0057829, -0.0009139, -0.0038145, 0.0038145)
7: (-0.0013120, 0.0053191, -0.0013120, 0.0053191, -0.0051950, 0.0051950)
8: (0.9882897, 0.9929608, 0.9882897, 0.9929608, -0.0036595, 0.0036595)
9: (-0.0094975, -0.0052574, -0.0094975, -0.0052574, -0.0033218, 0.0033218)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.68 + 1.95 = 3.63 seconds
status: Status.VERIFIED
relational distance
Output dim: 8, lower bound: -0.0019393, upper bound: 0.0019393
