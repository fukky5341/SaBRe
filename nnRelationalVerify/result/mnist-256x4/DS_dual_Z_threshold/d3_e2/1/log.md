## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 4.731830941972703


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.8503866, 1.3866532, -1.8503866, 1.3866532, -3.2370398, 3.2370398)
1: (-1.4591178, 1.3697687, -1.4591178, 1.3697687, -2.8288865, 2.8288865)
2: (-1.7938340, 1.9670236, -1.7938340, 1.9670236, -3.7608576, 3.7608576)
3: (-1.8323436, 1.2170542, -1.8323436, 1.2170542, -3.0493979, 3.0493979)
4: (-1.9638085, 1.4680431, -1.9638085, 1.4680431, -3.4318516, 3.4318516)
5: (-1.5762671, 1.4573678, -1.5762671, 1.4573678, -3.0336349, 3.0336349)
6: (-1.4516947, 1.6586541, -1.4516947, 1.6586541, -3.1103487, 3.1103487)
7: (-1.8006461, 1.7152629, -1.8006461, 1.7152629, -3.5159090, 3.5159090)
8: (-2.4382658, 1.6120684, -2.4382658, 1.6120684, -4.0503340, 4.0503340)
9: (-1.4212788, 1.6854837, -1.4212788, 1.6854837, -3.1067624, 3.1067624)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.12 + 3.63 = 5.75 seconds
status: Status.VERIFIED
relational distance
Output dim: 8, lower bound: -3.7040631, upper bound: 3.7040631
