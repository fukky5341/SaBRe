## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00039634


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0169902, 0.0174756, 0.0169902, 0.0174756, -0.0002377, 0.0002377)
1: (-0.0005385, -0.0002090, -0.0005385, -0.0002090, -0.0001728, 0.0001728)
2: (0.0038202, 0.0039790, 0.0038202, 0.0039790, -0.0000763, 0.0000763)
3: (0.0017808, 0.0020670, 0.0017808, 0.0020670, -0.0001440, 0.0001440)
4: (-0.0040421, -0.0036179, -0.0040421, -0.0036179, -0.0001666, 0.0001666)
5: (-0.0000208, 0.0001746, -0.0000208, 0.0001746, -0.0001049, 0.0001049)
6: (-0.0037858, -0.0029894, -0.0037858, -0.0029894, -0.0003372, 0.0003372)
7: (-0.0194559, -0.0170310, -0.0194559, -0.0170310, -0.0009546, 0.0009546)
8: (0.9774719, 0.9795880, 0.9774719, 0.9795880, -0.0008617, 0.0008617)
9: (0.0034993, 0.0050782, 0.0034993, 0.0050782, -0.0006245, 0.0006245)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.34 + 1.17 = 2.51 seconds
status: Status.ADV_EXAMPLE
