## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00226296


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0023682, -0.0006286, -0.0023682, -0.0006286, -0.0017396, 0.0017396)
1: (0.9983194, 1.0017174, 0.9983194, 1.0017174, -0.0025530, 0.0025530)
2: (-0.0013367, 0.0013146, -0.0013367, 0.0013146, -0.0026513, 0.0026513)
3: (0.0010514, 0.0020408, 0.0010514, 0.0020408, -0.0006985, 0.0006985)
4: (-0.0012629, 0.0007723, -0.0012629, 0.0007723, -0.0019120, 0.0019120)
5: (-0.0002988, 0.0018854, -0.0002988, 0.0018854, -0.0021842, 0.0021842)
6: (-0.0003165, 0.0012673, -0.0003165, 0.0012673, -0.0015838, 0.0015838)
7: (-0.0049697, -0.0030357, -0.0049697, -0.0030357, -0.0016196, 0.0016196)
8: (-0.0091463, -0.0042843, -0.0091463, -0.0042843, -0.0043218, 0.0043218)
9: (0.0021764, 0.0050287, 0.0021764, 0.0050287, -0.0028523, 0.0028523)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.54 + 1.41 = 2.94 seconds
status: Status.VERIFIED
relational distance
Output dim: 1, lower bound: -0.0021296, upper bound: 0.0021296
