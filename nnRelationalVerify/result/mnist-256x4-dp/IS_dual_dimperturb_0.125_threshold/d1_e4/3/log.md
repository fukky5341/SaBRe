## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 8.44e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0016857, -0.0015280, -0.0016857, -0.0015280, -0.0000646, 0.0000646)
1: (-0.0085883, -0.0081880, -0.0085883, -0.0081880, -0.0001639, 0.0001639)
2: (0.0297018, 0.0299502, 0.0297018, 0.0299502, -0.0001017, 0.0001017)
3: (0.0036380, 0.0041017, 0.0036380, 0.0041017, -0.0001899, 0.0001899)
4: (-0.0076288, -0.0072216, -0.0076288, -0.0072216, -0.0001668, 0.0001668)
5: (0.0108486, 0.0110028, 0.0108486, 0.0110028, -0.0000632, 0.0000632)
6: (0.0050052, 0.0055937, 0.0050052, 0.0055937, -0.0002410, 0.0002410)
7: (0.9815616, 0.9819735, 0.9815616, 0.9819735, -0.0001687, 0.0001687)
8: (-0.0063330, -0.0058915, -0.0063330, -0.0058915, -0.0001808, 0.0001808)
9: (-0.0011079, -0.0008163, -0.0011079, -0.0008163, -0.0001195, 0.0001195)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.59 + 1.21 = 2.80 seconds
status: Status.ADV_EXAMPLE
