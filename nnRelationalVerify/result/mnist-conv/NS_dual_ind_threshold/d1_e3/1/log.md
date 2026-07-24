## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.14719061993964846


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.3936348, -5.8157001, -6.3936348, -5.8157001, -0.2194817, 0.2194817)
1: (-10.5362320, -9.8813667, -10.5362320, -9.8813667, -0.2550082, 0.2550081)
2: (4.2153654, 4.7343335, 4.2153654, 4.7343335, -0.2240150, 0.2240148)
3: (-4.9839268, -4.3914723, -4.9839268, -4.3914723, -0.2422287, 0.2422287)
4: (-5.7332196, -5.1679640, -5.7332196, -5.1679640, -0.2779112, 0.2779113)
5: (-8.7329502, -8.1154823, -8.7329502, -8.1154823, -0.2431337, 0.2431337)
6: (-9.0211830, -8.3237667, -9.0211830, -8.3237667, -0.2398095, 0.2398094)
7: (-4.7985044, -4.3397350, -4.7985044, -4.3397350, -0.2155526, 0.2155528)
8: (1.2806940, 1.7087145, 1.2806940, 1.7087145, -0.1763926, 0.1763926)
9: (-10.4673672, -9.9915791, -10.4673672, -9.9915791, -0.1951673, 0.1951673)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.11 + 32.73 = 55.84 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.1201653, upper bound: 0.1201652
