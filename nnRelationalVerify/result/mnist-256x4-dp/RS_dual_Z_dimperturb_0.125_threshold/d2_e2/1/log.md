## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.03610424


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0040544, 0.0480820, 0.0040544, 0.0480820, -0.0261466, 0.0261466)
1: (0.0033234, 0.0043187, 0.0033234, 0.0043187, -0.0008669, 0.0008669)
2: (0.0192276, 0.0331824, 0.0192276, 0.0331824, -0.0083144, 0.0083144)
3: (0.0321743, 0.0600149, 0.0321743, 0.0600149, -0.0163273, 0.0163273)
4: (-0.0120640, -0.0049915, -0.0120640, -0.0049915, -0.0049366, 0.0049366)
5: (0.0286290, 0.0441061, 0.0286290, 0.0441061, -0.0090301, 0.0090301)
6: (-0.0034074, 0.0388545, -0.0034074, 0.0388545, -0.0247857, 0.0247857)
7: (-0.0066435, -0.0062859, -0.0066435, -0.0062859, -0.0003576, 0.0003576)
8: (0.7327499, 0.8585346, 0.7327499, 0.8585346, -0.0736309, 0.0736309)
9: (0.0767687, 0.0889769, 0.0767687, 0.0889769, -0.0072712, 0.0072712)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 1.16 = 2.48 seconds
status: Status.ADV_EXAMPLE
