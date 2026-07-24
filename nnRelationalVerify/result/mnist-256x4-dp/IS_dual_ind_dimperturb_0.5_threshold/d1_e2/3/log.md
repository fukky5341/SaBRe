## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00077259


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0034288, 0.0024949, -0.0034288, 0.0024949, -0.0059236, 0.0059236)
1: (-0.0044674, -0.0020943, -0.0044674, -0.0020943, -0.0023731, 0.0023731)
2: (0.0308085, 0.0345008, 0.0308085, 0.0345008, -0.0036923, 0.0036923)
3: (-0.0039646, 0.0010051, -0.0039646, 0.0010051, -0.0040798, 0.0040798)
4: (-0.0028823, 0.0004565, -0.0028823, 0.0004565, -0.0033388, 0.0033388)
5: (0.0095521, 0.0148472, 0.0095521, 0.0148472, -0.0052951, 0.0052951)
6: (-0.0060139, -0.0016294, -0.0060139, -0.0016294, -0.0043846, 0.0043846)
7: (0.9744390, 0.9775204, 0.9744390, 0.9775204, -0.0030814, 0.0030814)
8: (-0.0172074, -0.0024366, -0.0172074, -0.0024366, -0.0147707, 0.0147707)
9: (-0.0025926, 0.0059574, -0.0025926, 0.0059574, -0.0085500, 0.0085500)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.20 + 1.30 = 2.50 seconds
status: Status.ADV_EXAMPLE
