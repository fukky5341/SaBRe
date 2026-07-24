## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00027306


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9945141, 0.9959671, 0.9945141, 0.9959671, -0.0009830, 0.0009830)
1: (-0.0026309, -0.0022689, -0.0026309, -0.0022689, -0.0002449, 0.0002449)
2: (0.0019697, 0.0038884, 0.0019697, 0.0038884, -0.0012980, 0.0012980)
3: (-0.0030429, -0.0021697, -0.0030429, -0.0021697, -0.0005908, 0.0005908)
4: (0.0009091, 0.0012805, 0.0009091, 0.0012805, -0.0002512, 0.0002512)
5: (0.0014368, 0.0038500, 0.0014368, 0.0038500, -0.0016325, 0.0016325)
6: (0.0005637, 0.0011761, 0.0005637, 0.0011761, -0.0004144, 0.0004144)
7: (-0.0016793, -0.0000946, -0.0016793, -0.0000946, -0.0010721, 0.0010721)
8: (-0.0004473, 0.0003861, -0.0004473, 0.0003861, -0.0005638, 0.0005638)
9: (-0.0023116, -0.0013452, -0.0023116, -0.0013452, -0.0006537, 0.0006537)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.79 + 1.29 = 3.08 seconds
status: Status.ADV_EXAMPLE
