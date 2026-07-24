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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9541693, 0.9915420, 0.9541693, 0.9915420, -0.0373727, 0.0373727)
1: (-0.0069474, -0.0029452, -0.0069474, -0.0029452, -0.0040022, 0.0040022)
2: (0.0078130, 0.0172980, 0.0078130, 0.0172980, -0.0092345, 0.0092345)
3: (-0.0087205, -0.0034271, -0.0087205, -0.0034271, -0.0052934, 0.0052934)
4: (0.0008084, 0.0052801, 0.0008084, 0.0052801, -0.0044717, 0.0044717)
5: (0.0087861, 0.0406698, 0.0087861, 0.0406698, -0.0318837, 0.0318837)
6: (-0.0049914, 0.0011262, -0.0049914, 0.0011262, -0.0061176, 0.0061176)
7: (-0.0137104, -0.0049207, -0.0137104, -0.0049207, -0.0087896, 0.0087896)
8: (-0.0060936, 0.0069980, -0.0060936, 0.0069980, -0.0130916, 0.0130916)
9: (0.0006314, 0.0076575, 0.0006314, 0.0076575, -0.0070261, 0.0070261)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.89 + 2.77 = 4.66 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.0336163, upper bound: 0.0336163
