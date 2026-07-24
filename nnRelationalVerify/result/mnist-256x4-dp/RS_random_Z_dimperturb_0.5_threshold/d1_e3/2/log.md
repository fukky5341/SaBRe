## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00109205


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0043741, -0.0041121, -0.0043741, -0.0041121, -0.0002029, 0.0002029)
1: (0.0025164, 0.0039674, 0.0025164, 0.0039674, -0.0011237, 0.0011237)
2: (0.0061024, 0.0093442, 0.0061024, 0.0093442, -0.0025104, 0.0025104)
3: (0.0033967, 0.0047628, 0.0033967, 0.0047628, -0.0010579, 0.0010579)
4: (1.0099282, 1.0152279, 1.0099282, 1.0152279, -0.0041042, 0.0041042)
5: (0.0041728, 0.0052039, 0.0041728, 0.0052039, -0.0007984, 0.0007984)
6: (-0.0125150, -0.0111733, -0.0125150, -0.0111733, -0.0010390, 0.0010390)
7: (-0.0103998, -0.0102286, -0.0103998, -0.0102286, -0.0001325, 0.0001325)
8: (-0.0031343, -0.0022072, -0.0031343, -0.0022072, -0.0007179, 0.0007179)
9: (-0.0071209, -0.0024800, -0.0071209, -0.0024800, -0.0035939, 0.0035939)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.36 + 1.44 = 2.80 seconds
status: Status.ADV_EXAMPLE
