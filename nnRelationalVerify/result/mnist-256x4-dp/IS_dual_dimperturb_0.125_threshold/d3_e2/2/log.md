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
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041416, -0.0041351, -0.0041416, -0.0041351, -0.0000049, 0.0000049)
1: (-0.0077595, -0.0075141, -0.0077595, -0.0075141, -0.0001829, 0.0001829)
2: (0.9671518, 0.9674463, 0.9671518, 0.9674463, -0.0002195, 0.0002195)
3: (0.0040230, 0.0061952, 0.0040230, 0.0061952, -0.0016193, 0.0016193)
4: (-0.0011642, -0.0009990, -0.0011642, -0.0009990, -0.0001232, 0.0001232)
5: (0.0160937, 0.0162607, 0.0160937, 0.0162607, -0.0001245, 0.0001245)
6: (0.0039976, 0.0040788, 0.0039976, 0.0040788, -0.0000605, 0.0000605)
7: (-0.0093838, -0.0088209, -0.0093838, -0.0088209, -0.0004197, 0.0004197)
8: (0.0092845, 0.0097311, 0.0092845, 0.0097311, -0.0003329, 0.0003329)
9: (0.0144237, 0.0152269, 0.0144237, 0.0152269, -0.0005988, 0.0005988)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.58 + 1.18 = 2.76 seconds
status: Status.ADV_EXAMPLE
