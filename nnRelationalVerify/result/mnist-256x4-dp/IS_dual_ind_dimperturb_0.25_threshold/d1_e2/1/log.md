## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00013584


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0038049, -0.0024716, -0.0038049, -0.0024716, -0.0013333, 0.0013333)
1: (0.0054879, 0.0064258, 0.0054879, 0.0064258, -0.0009378, 0.0009378)
2: (0.0109617, 0.0134673, 0.0109617, 0.0134673, -0.0021129, 0.0021129)
3: (-0.0041156, -0.0029889, -0.0041156, -0.0029889, -0.0011174, 0.0011174)
4: (0.0047581, 0.0051340, 0.0047581, 0.0051340, -0.0002450, 0.0002450)
5: (-0.0018793, -0.0010506, -0.0018793, -0.0010506, -0.0008286, 0.0008286)
6: (-0.0058014, -0.0053959, -0.0058014, -0.0053959, -0.0004055, 0.0004055)
7: (-0.0030697, -0.0023150, -0.0030697, -0.0023150, -0.0007546, 0.0007546)
8: (-0.0033837, -0.0017261, -0.0033837, -0.0017261, -0.0016576, 0.0016576)
9: (1.0004562, 1.0005955, 1.0004562, 1.0005955, -0.0001392, 0.0001392)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.22 + 1.37 = 2.58 seconds
status: Status.VERIFIED
relational distance
Output dim: 9, lower bound: -0.0001278, upper bound: 0.0001278
