## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0004916


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0002038, 0.0008159, 0.0002038, 0.0008159, -0.0004339, 0.0004339)
1: (0.9941515, 0.9954478, 0.9941515, 0.9954478, -0.0009189, 0.0009189)
2: (-0.0079040, -0.0075978, -0.0079040, -0.0075978, -0.0002170, 0.0002170)
3: (0.0029430, 0.0037088, 0.0029430, 0.0037088, -0.0005428, 0.0005428)
4: (0.0028339, 0.0038320, 0.0028339, 0.0038320, -0.0007075, 0.0007075)
5: (0.0038451, 0.0052957, 0.0038451, 0.0052957, -0.0010282, 0.0010282)
6: (-0.0010698, 0.0002715, -0.0010698, 0.0002715, -0.0009508, 0.0009508)
7: (-0.0075673, -0.0069468, -0.0075673, -0.0069468, -0.0004398, 0.0004398)
8: (0.0080817, 0.0081634, 0.0080817, 0.0081634, -0.0000579, 0.0000579)
9: (-0.0032414, -0.0023554, -0.0032414, -0.0023554, -0.0006281, 0.0006281)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.34 + 1.18 = 2.51 seconds
status: Status.ADV_EXAMPLE
