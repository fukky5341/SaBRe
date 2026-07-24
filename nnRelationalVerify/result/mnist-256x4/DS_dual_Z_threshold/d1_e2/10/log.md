## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.004873019728776953


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0059522, -0.0038062, -0.0059522, -0.0038062, -0.0021460, 0.0021460)
1: (-0.0105666, -0.0091769, -0.0105666, -0.0091769, -0.0013896, 0.0013896)
2: (0.9630674, 0.9695855, 0.9630674, 0.9695855, -0.0065181, 0.0065181)
3: (-0.0190312, -0.0085235, -0.0190312, -0.0085235, -0.0104320, 0.0104320)
4: (-0.0013857, 0.0013758, -0.0013857, 0.0013758, -0.0022319, 0.0022319)
5: (0.0163889, 0.0199546, 0.0163889, 0.0199546, -0.0035658, 0.0035658)
6: (-0.0010723, 0.0057559, -0.0010723, 0.0057559, -0.0068282, 0.0068282)
7: (-0.0068919, 0.0009421, -0.0068919, 0.0009421, -0.0067835, 0.0067835)
8: (0.0109214, 0.0136058, 0.0109214, 0.0136058, -0.0026844, 0.0026844)
9: (0.0198666, 0.0230548, 0.0198666, 0.0230548, -0.0031882, 0.0031882)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.15 + 2.35 = 3.50 seconds
status: Status.VERIFIED
relational distance
Output dim: 2, lower bound: -0.0043920, upper bound: 0.0043920
