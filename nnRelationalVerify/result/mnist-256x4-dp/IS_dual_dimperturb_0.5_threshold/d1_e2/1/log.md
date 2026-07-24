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
Threshold: 0.00045437


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0042098, -0.0015174, -0.0042098, -0.0015174, -0.0026925, 0.0026925)
1: (0.0048260, 0.0067194, 0.0048260, 0.0067194, -0.0018934, 0.0018934)
2: (0.0101992, 0.0152668, 0.0101992, 0.0152668, -0.0045408, 0.0045408)
3: (-0.0049077, -0.0026418, -0.0049077, -0.0026418, -0.0022659, 0.0022659)
4: (0.0044935, 0.0052492, 0.0044935, 0.0052492, -0.0007406, 0.0007406)
5: (-0.0024870, -0.0007944, -0.0024870, -0.0007944, -0.0016926, 0.0016926)
6: (-0.0060896, -0.0052723, -0.0060896, -0.0052723, -0.0008173, 0.0008173)
7: (-0.0033278, -0.0017784, -0.0033278, -0.0017784, -0.0015494, 0.0015494)
8: (-0.0045758, -0.0012211, -0.0045758, -0.0012211, -0.0033547, 0.0033547)
9: (1.0004152, 1.0010928, 1.0004152, 1.0010928, -0.0006776, 0.0006776)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.30 + 1.39 = 2.70 seconds
status: Status.ADV_EXAMPLE
