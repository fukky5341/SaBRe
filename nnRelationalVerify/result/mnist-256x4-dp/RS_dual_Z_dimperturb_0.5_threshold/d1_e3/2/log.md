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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0043707, -0.0041098, -0.0043707, -0.0041098, -0.0001951, 0.0001951)
1: (0.0025041, 0.0039483, 0.0025041, 0.0039483, -0.0010803, 0.0010803)
2: (0.0061453, 0.0093718, 0.0061453, 0.0093718, -0.0024136, 0.0024136)
3: (0.0033850, 0.0047447, 0.0033850, 0.0047447, -0.0010171, 0.0010171)
4: (1.0098829, 1.0151579, 1.0098829, 1.0151579, -0.0039459, 0.0039459)
5: (0.0041640, 0.0051902, 0.0041640, 0.0051902, -0.0007676, 0.0007676)
6: (-0.0124973, -0.0111618, -0.0124973, -0.0111618, -0.0009990, 0.0009990)
7: (-0.0103975, -0.0102272, -0.0103975, -0.0102272, -0.0001274, 0.0001274)
8: (-0.0031422, -0.0022195, -0.0031422, -0.0022195, -0.0006902, 0.0006902)
9: (-0.0070596, -0.0024404, -0.0070596, -0.0024404, -0.0034554, 0.0034554)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.27 + 1.43 = 2.71 seconds
status: Status.ADV_EXAMPLE
