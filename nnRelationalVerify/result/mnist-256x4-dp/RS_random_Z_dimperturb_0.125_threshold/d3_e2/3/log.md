## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00990792


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0116810, 0.0018521, -0.0116810, 0.0018521, -0.0110173, 0.0110174)
1: (-0.0123875, -0.0015844, -0.0123875, -0.0015844, -0.0104791, 0.0104791)
2: (0.0444152, 0.0502930, 0.0444152, 0.0502930, -0.0058779, 0.0058779)
3: (0.0083136, 0.0301541, 0.0083136, 0.0301541, -0.0174103, 0.0174103)
4: (-0.0039746, 0.0004122, -0.0039746, 0.0004122, -0.0043868, 0.0043868)
5: (0.0111373, 0.0135396, 0.0111373, 0.0135396, -0.0024023, 0.0024023)
6: (-0.0272658, -0.0138228, -0.0272658, -0.0138228, -0.0134430, 0.0134430)
7: (0.9170767, 0.9543704, 0.9170767, 0.9543704, -0.0372937, 0.0372937)
8: (0.0005540, 0.0176296, 0.0005540, 0.0176296, -0.0170756, 0.0170756)
9: (-0.0083957, -0.0026417, -0.0083957, -0.0026417, -0.0057540, 0.0057540)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.66 + 1.24 = 2.90 seconds
status: Status.ADV_EXAMPLE
