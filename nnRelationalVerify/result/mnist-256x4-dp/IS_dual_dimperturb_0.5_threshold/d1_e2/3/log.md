## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00077259


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0033130, 0.0022553, -0.0033130, 0.0022553, -0.0055683, 0.0055683)
1: (-0.0044215, -0.0021909, -0.0044215, -0.0021909, -0.0022306, 0.0022306)
2: (0.0308756, 0.0343452, 0.0308756, 0.0343452, -0.0034696, 0.0034696)
3: (-0.0037851, 0.0008796, -0.0037851, 0.0008796, -0.0038408, 0.0038408)
4: (-0.0027496, 0.0003362, -0.0027496, 0.0003362, -0.0030858, 0.0030858)
5: (0.0096484, 0.0146240, 0.0096484, 0.0146240, -0.0049757, 0.0049757)
6: (-0.0056510, -0.0017762, -0.0056510, -0.0017762, -0.0038748, 0.0038748)
7: (0.9746667, 0.9774812, 0.9746667, 0.9774812, -0.0028145, 0.0028145)
8: (-0.0165848, -0.0027051, -0.0165848, -0.0027051, -0.0138797, 0.0138797)
9: (-0.0024372, 0.0055971, -0.0024372, 0.0055971, -0.0080342, 0.0080342)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.25 + 1.31 = 2.57 seconds
status: Status.ADV_EXAMPLE
