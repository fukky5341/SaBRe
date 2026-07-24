## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.03428451


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0103531, 0.0039234, -0.0103531, 0.0039234, -0.0137250, 0.0137250)
1: (-0.0024647, 0.0073345, -0.0024647, 0.0073345, -0.0097991, 0.0097991)
2: (0.0053559, 0.0408509, 0.0053559, 0.0408509, -0.0349798, 0.0349798)
3: (-0.0065995, 0.0110471, -0.0065995, 0.0110471, -0.0171415, 0.0171415)
4: (-0.0104670, 0.0195583, -0.0104670, 0.0195583, -0.0300253, 0.0300253)
5: (0.0011354, 0.0117572, 0.0011354, 0.0117572, -0.0106219, 0.0106219)
6: (0.0007549, 0.0124232, 0.0007549, 0.0124232, -0.0116683, 0.0116683)
7: (-0.0334717, -0.0013318, -0.0334717, -0.0013318, -0.0281047, 0.0281047)
8: (0.9521940, 1.0199754, 0.9521940, 1.0199754, -0.0677814, 0.0677814)
9: (-0.0090175, 0.0094358, -0.0090175, 0.0094358, -0.0184533, 0.0184533)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.54 + 1.88 = 3.42 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0353136, upper bound: 0.0353136

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 250

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 78

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0340652, upper bound: 0.0339291
time: 0.99 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0339267, upper bound: 0.0339267
time: 0.99 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.13 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 2.13
Output dim: 8, lower bound: -0.0340652, upper bound: 0.0339291
IS_A2, status: Status.VERIFIED, split count: 1, time: 2.13
Output dim: 8, lower bound: -0.0339267, upper bound: 0.0339267

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.42 + 2.13 = 5.55 seconds
