## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 6.64e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040688, -0.0040644, -0.0040688, -0.0040644, -0.0000020, 0.0000020)
1: (-0.0050326, -0.0048660, -0.0050326, -0.0048660, -0.0000748, 0.0000748)
2: (0.9704241, 0.9706241, 0.9704241, 0.9706241, -0.0000898, 0.0000898)
3: (0.0281593, 0.0296340, 0.0281593, 0.0296340, -0.0006624, 0.0006624)
4: (-0.0029469, -0.0028347, -0.0029469, -0.0028347, -0.0000504, 0.0000504)
5: (0.0142920, 0.0144054, 0.0142920, 0.0144054, -0.0000509, 0.0000509)
6: (0.0049000, 0.0049552, 0.0049000, 0.0049552, -0.0000248, 0.0000248)
7: (-0.0154582, -0.0150760, -0.0154582, -0.0150760, -0.0001717, 0.0001717)
8: (0.0044654, 0.0047686, 0.0044654, 0.0047686, -0.0001362, 0.0001362)
9: (0.0057560, 0.0063014, 0.0057560, 0.0063014, -0.0002450, 0.0002450)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.26 + 1.14 = 2.40 seconds
status: Status.ADV_EXAMPLE
