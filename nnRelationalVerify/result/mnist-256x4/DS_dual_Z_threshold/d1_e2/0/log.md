## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.004025293620874309


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0002245, 0.0017627, -0.0002245, 0.0017627, -0.0019872, 0.0019872)
1: (0.9920171, 0.9970278, 0.9920171, 0.9970278, -0.0050107, 0.0050107)
2: (-0.0086045, -0.0025493, -0.0086045, -0.0025493, -0.0057238, 0.0057238)
3: (0.0025766, 0.0046549, 0.0025766, 0.0046549, -0.0020783, 0.0020783)
4: (0.0013672, 0.0052175, 0.0013672, 0.0052175, -0.0038503, 0.0038503)
5: (0.0031408, 0.0080109, 0.0031408, 0.0080109, -0.0048701, 0.0048701)
6: (-0.0021078, 0.0000522, -0.0021078, 0.0000522, -0.0021600, 0.0021600)
7: (-0.0093804, -0.0058737, -0.0093804, -0.0058737, -0.0035067, 0.0035067)
8: (-0.0014898, 0.0095568, -0.0014898, 0.0095568, -0.0109517, 0.0109517)
9: (-0.0058226, 0.0005370, -0.0058226, 0.0005370, -0.0063596, 0.0063596)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.25 + 2.95 = 4.21 seconds
status: Status.VERIFIED
relational distance
Output dim: 1, lower bound: -0.0032357, upper bound: 0.0032357
