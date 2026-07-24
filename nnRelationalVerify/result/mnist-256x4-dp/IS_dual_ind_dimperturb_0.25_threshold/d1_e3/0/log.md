## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000280174


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (1.0033352, 1.0042753, 1.0033352, 1.0042753, -0.0006104, 0.0006104)
1: (-0.0004329, -0.0001987, -0.0004329, -0.0001987, -0.0001521, 0.0001521)
2: (-0.0090011, -0.0077599, -0.0090011, -0.0077599, -0.0008060, 0.0008060)
3: (0.0022588, 0.0028238, 0.0022588, 0.0028238, -0.0003668, 0.0003668)
4: (-0.0012143, -0.0009740, -0.0012143, -0.0009740, -0.0001560, 0.0001560)
5: (-0.0123616, -0.0108005, -0.0123616, -0.0108005, -0.0010137, 0.0010137)
6: (0.0042821, 0.0046783, 0.0042821, 0.0046783, -0.0002573, 0.0002573)
7: (0.0079415, 0.0089667, 0.0079415, 0.0089667, -0.0006657, 0.0006657)
8: (0.0046122, 0.0051513, 0.0046122, 0.0051513, -0.0003501, 0.0003501)
9: (-0.0078371, -0.0072119, -0.0078371, -0.0072119, -0.0004059, 0.0004059)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.26 + 1.34 = 2.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0002897, upper bound: 0.0002897

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002642, upper bound: 0.0002264
time: 0.55 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0002642, upper bound: 0.0002642
time: 0.52 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.20 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 1.20
Output dim: 0, lower bound: -0.0002642, upper bound: 0.0002264
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.20
Output dim: 0, lower bound: -0.0002642, upper bound: 0.0002642

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.60 + 1.20 = 3.80 seconds
