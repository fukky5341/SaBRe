## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00070371


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=21, inp2_unstable=21, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0058437, 0.0064668, 0.0058437, 0.0064668, -0.0006232, 0.0006232)
1: (-0.0006336, 0.0006077, -0.0006336, 0.0006077, -0.0012414, 0.0012414)
2: (0.0120259, 0.0218046, 0.0120259, 0.0218046, -0.0097787, 0.0097787)
3: (-0.0044562, -0.0035867, -0.0044562, -0.0035867, -0.0008695, 0.0008695)
4: (-0.0002845, 0.0039343, -0.0002845, 0.0039343, -0.0042188, 0.0042188)
5: (-0.0011071, -0.0004491, -0.0011071, -0.0004491, -0.0006579, 0.0006579)
6: (0.9910547, 0.9922835, 0.9910547, 0.9922835, -0.0012288, 0.0012288)
7: (-0.0139120, -0.0062610, -0.0139120, -0.0062610, -0.0062561, 0.0062561)
8: (-0.0035758, -0.0009732, -0.0035758, -0.0009732, -0.0026027, 0.0026027)
9: (-0.0053868, -0.0005837, -0.0053868, -0.0005837, -0.0048030, 0.0048030)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.65 + 2.17 = 3.81 seconds
status: Status.VERIFIED
relational distance
Output dim: 6, lower bound: -0.0005925, upper bound: 0.0005925
