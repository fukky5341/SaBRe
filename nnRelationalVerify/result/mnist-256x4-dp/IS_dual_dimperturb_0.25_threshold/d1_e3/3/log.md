## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00603328


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0030033, -0.0019358, -0.0030033, -0.0019358, -0.0009339, 0.0009339)
1: (0.0227097, 0.0290278, 0.0227097, 0.0290278, -0.0030330, 0.0030330)
2: (0.0226636, 0.0268505, 0.0226636, 0.0268505, -0.0022192, 0.0022192)
3: (0.0102953, 0.0149678, 0.0102953, 0.0149678, -0.0031511, 0.0031511)
4: (-0.0153568, -0.0104891, -0.0153568, -0.0104891, -0.0033699, 0.0033699)
5: (0.0175159, 0.0231938, 0.0175159, 0.0231938, -0.0037961, 0.0037961)
6: (0.0083575, 0.0127516, 0.0083575, 0.0127516, -0.0031035, 0.0031035)
7: (-0.0200350, -0.0155195, -0.0200350, -0.0155195, -0.0029636, 0.0029636)
8: (0.0122615, 0.0168052, 0.0122615, 0.0168052, -0.0029520, 0.0029520)
9: (0.9114316, 0.9336104, 0.9114316, 0.9336104, -0.0138971, 0.0138971)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.27 + 1.34 = 2.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0079825, upper bound: 0.0079825

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.ADV_EXAMPLE
time: 0.51 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.ADV_EXAMPLE
time: 0.42 seconds

## IS Result
status: Status.ADV_EXAMPLE
execution time: (base) + (is) = 2.62 + 1.06 = 3.68 seconds
