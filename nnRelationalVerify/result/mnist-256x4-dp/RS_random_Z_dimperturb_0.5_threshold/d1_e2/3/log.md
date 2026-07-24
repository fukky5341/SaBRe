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
0: (-0.0032302, 0.0022906, -0.0032302, 0.0022906, -0.0055208, 0.0055208)
1: (-0.0043887, -0.0021767, -0.0043887, -0.0021767, -0.0022120, 0.0022120)
2: (0.0309236, 0.0343681, 0.0309236, 0.0343681, -0.0034445, 0.0034445)
3: (-0.0038116, 0.0007898, -0.0038116, 0.0007898, -0.0037510, 0.0037510)
4: (-0.0027692, 0.0002501, -0.0027692, 0.0002501, -0.0030192, 0.0030192)
5: (0.0097172, 0.0146569, 0.0097172, 0.0146569, -0.0049397, 0.0049397)
6: (-0.0053913, -0.0017546, -0.0053913, -0.0017546, -0.0036367, 0.0036367)
7: (0.9748294, 0.9774531, 0.9748294, 0.9774531, -0.0026236, 0.0026236)
8: (-0.0166765, -0.0028971, -0.0166765, -0.0028971, -0.0137793, 0.0137793)
9: (-0.0023260, 0.0056501, -0.0023260, 0.0056501, -0.0079761, 0.0079761)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.13 + 1.27 = 2.40 seconds
status: Status.ADV_EXAMPLE
