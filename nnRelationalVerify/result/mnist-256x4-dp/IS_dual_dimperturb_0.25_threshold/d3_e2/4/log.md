## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00117945


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041089, -0.0038565, -0.0041089, -0.0038565, -0.0001596, 0.0001596)
1: (0.0011013, 0.0024988, 0.0011013, 0.0024988, -0.0008835, 0.0008835)
2: (0.0093835, 0.0125057, 0.0093835, 0.0125057, -0.0019739, 0.0019739)
3: (0.0020644, 0.0033801, 0.0020644, 0.0033801, -0.0008318, 0.0008318)
4: (1.0047594, 1.0098640, 1.0047594, 1.0098640, -0.0032270, 0.0032270)
5: (0.0031673, 0.0041603, 0.0031673, 0.0041603, -0.0006278, 0.0006278)
6: (-0.0111570, -0.0098647, -0.0111570, -0.0098647, -0.0008170, 0.0008170)
7: (-0.0102265, -0.0100617, -0.0102265, -0.0100617, -0.0001042, 0.0001042)
8: (-0.0040384, -0.0031455, -0.0040384, -0.0031455, -0.0005645, 0.0005645)
9: (-0.0024237, 0.0020462, -0.0024237, 0.0020462, -0.0028258, 0.0028258)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.47 + 1.31 = 2.78 seconds
status: Status.ADV_EXAMPLE
