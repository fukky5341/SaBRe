## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0061821


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9834952, 0.9898838, 0.9834952, 0.9898838, -0.0063886, 0.0063886)
1: (-0.0044923, -0.0037847, -0.0044923, -0.0037847, -0.0007077, 0.0007077)
2: (0.0100026, 0.0137530, 0.0100026, 0.0137530, -0.0037504, 0.0037504)
3: (-0.0075917, -0.0058259, -0.0075917, -0.0058259, -0.0017658, 0.0017658)
4: (0.0024639, 0.0034442, 0.0024639, 0.0034442, -0.0009803, 0.0009803)
5: (0.0115401, 0.0188929, 0.0115401, 0.0188929, -0.0073528, 0.0073528)
6: (-0.0025854, -0.0013882, -0.0025854, -0.0013882, -0.0011972, 0.0011972)
7: (-0.0098268, -0.0067293, -0.0098268, -0.0067293, -0.0030976, 0.0030976)
8: (-0.0047320, -0.0031030, -0.0047320, -0.0031030, -0.0016290, 0.0016290)
9: (0.0017342, 0.0036231, 0.0017342, 0.0036231, -0.0018889, 0.0018889)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.15 + 1.69 = 2.85 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -0.0050496, upper bound: 0.0050496
