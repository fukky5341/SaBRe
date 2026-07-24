## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00473928


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0093721, -0.0031455, -0.0093721, -0.0031455, -0.0038080, 0.0038080)
1: (-0.0050318, -0.0043366, -0.0050318, -0.0043366, -0.0006952, 0.0006952)
2: (0.0327918, 0.0489016, 0.0327918, 0.0489016, -0.0107932, 0.0107932)
3: (0.0007307, 0.0108153, 0.0007307, 0.0108153, -0.0047913, 0.0047913)
4: (-0.0040972, -0.0022615, -0.0040972, -0.0022615, -0.0018357, 0.0018357)
5: (0.0100650, 0.0117579, 0.0100650, 0.0117579, -0.0016930, 0.0016930)
6: (-0.0167579, -0.0021774, -0.0167579, -0.0021774, -0.0072247, 0.0072247)
7: (0.9552269, 0.9749365, 0.9552269, 0.9749365, -0.0197096, 0.0197096)
8: (-0.0055113, 0.0004637, -0.0055113, 0.0004637, -0.0059751, 0.0059751)
9: (-0.0037377, -0.0009895, -0.0037377, -0.0009895, -0.0027482, 0.0027482)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.41 + 1.24 = 2.66 seconds
status: Status.ADV_EXAMPLE
