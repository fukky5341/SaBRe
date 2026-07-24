## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 8.58e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041421, -0.0041352, -0.0041421, -0.0041352, -0.0000040, 0.0000040)
1: (-0.0077759, -0.0075175, -0.0077759, -0.0075175, -0.0001507, 0.0001507)
2: (0.9671320, 0.9674422, 0.9671320, 0.9674422, -0.0001809, 0.0001809)
3: (0.0038771, 0.0061648, 0.0038771, 0.0061648, -0.0013341, 0.0013341)
4: (-0.0011619, -0.0009879, -0.0011619, -0.0009879, -0.0001015, 0.0001015)
5: (0.0160960, 0.0162719, 0.0160960, 0.0162719, -0.0001025, 0.0001025)
6: (0.0039921, 0.0040777, 0.0039921, 0.0040777, -0.0000499, 0.0000499)
7: (-0.0093759, -0.0087830, -0.0093759, -0.0087830, -0.0003457, 0.0003457)
8: (0.0092907, 0.0097611, 0.0092907, 0.0097611, -0.0002743, 0.0002743)
9: (0.0144349, 0.0152809, 0.0144349, 0.0152809, -0.0004933, 0.0004933)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.64 + 1.20 = 2.85 seconds
status: Status.ADV_EXAMPLE
