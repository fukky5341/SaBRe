## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 6.64e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0040693, -0.0040640, -0.0040693, -0.0040640, -0.0000025, 0.0000025)
1: (-0.0050494, -0.0048534, -0.0050494, -0.0048534, -0.0000931, 0.0000931)
2: (0.9704039, 0.9706392, 0.9704039, 0.9706392, -0.0001117, 0.0001117)
3: (0.0280104, 0.0297456, 0.0280104, 0.0297456, -0.0008237, 0.0008237)
4: (-0.0029554, -0.0028234, -0.0029554, -0.0028234, -0.0000626, 0.0000626)
5: (0.0142834, 0.0144168, 0.0142834, 0.0144168, -0.0000633, 0.0000633)
6: (0.0048944, 0.0049593, 0.0048944, 0.0049593, -0.0000308, 0.0000308)
7: (-0.0154871, -0.0150374, -0.0154871, -0.0150374, -0.0002135, 0.0002135)
8: (0.0044424, 0.0047992, 0.0044424, 0.0047992, -0.0001694, 0.0001694)
9: (0.0057147, 0.0063564, 0.0057147, 0.0063564, -0.0003046, 0.0003046)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.30 + 1.17 = 2.46 seconds
status: Status.ADV_EXAMPLE
