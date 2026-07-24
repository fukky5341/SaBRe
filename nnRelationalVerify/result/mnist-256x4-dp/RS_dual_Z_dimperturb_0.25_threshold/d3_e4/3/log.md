## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.03662127


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=22, inp2_unstable=22, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9648181, 0.9910843, 0.9648181, 0.9910843, -0.0262662, 0.0262662)
1: (-0.0047639, -0.0031993, -0.0047639, -0.0031993, -0.0015646, 0.0015646)
2: (0.0084173, 0.0152321, 0.0084173, 0.0152321, -0.0066070, 0.0066070)
3: (-0.0084958, -0.0051043, -0.0084958, -0.0051043, -0.0033915, 0.0033915)
4: (0.0021570, 0.0049146, 0.0021570, 0.0049146, -0.0026379, 0.0026379)
5: (0.0095462, 0.0331009, 0.0095462, 0.0331009, -0.0235546, 0.0235546)
6: (-0.0030242, -0.0004096, -0.0030242, -0.0004096, -0.0026145, 0.0026145)
7: (-0.0109621, -0.0054199, -0.0109621, -0.0054199, -0.0055422, 0.0055422)
8: (-0.0053290, 0.0025744, -0.0053290, 0.0025744, -0.0079034, 0.0079034)
9: (0.0009358, 0.0045252, 0.0009358, 0.0045252, -0.0035895, 0.0035895)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.92 + 1.78 = 3.69 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.0234052, upper bound: 0.0234052
