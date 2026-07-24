## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0052569


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0083835, 0.0037988, -0.0083835, 0.0037988, -0.0121823, 0.0121823)
1: (0.0007282, 0.0097007, 0.0007282, 0.0097007, -0.0089725, 0.0089725)
2: (0.0023634, 0.0253543, 0.0023634, 0.0253543, -0.0204614, 0.0204614)
3: (-0.0092671, 0.0009077, -0.0092671, 0.0009077, -0.0101748, 0.0101748)
4: (0.0030388, 0.0064297, 0.0030388, 0.0064297, -0.0024941, 0.0024941)
5: (-0.0059873, 0.0017797, -0.0059873, 0.0017797, -0.0077670, 0.0077670)
6: (-0.0076793, -0.0040030, -0.0076793, -0.0040030, -0.0036763, 0.0036763)
7: (-0.0061346, 0.0016010, -0.0061346, 0.0016010, -0.0077356, 0.0077356)
8: (-0.0112792, 0.0039576, -0.0112792, 0.0039576, -0.0152368, 0.0152368)
9: (0.9999947, 1.0063156, 0.9999947, 1.0063156, -0.0063209, 0.0063209)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.93 + 2.32 = 3.25 seconds
status: Status.VERIFIED
relational distance
Output dim: 9, lower bound: -0.0058410, upper bound: 0.0058410
