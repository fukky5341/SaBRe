## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00365364


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0032614, -0.0005231, -0.0032614, -0.0005231, -0.0021864, 0.0021864)
1: (-0.0125869, -0.0056381, -0.0125869, -0.0056381, -0.0055483, 0.0055483)
2: (0.0272210, 0.0315321, 0.0272210, 0.0315321, -0.0034422, 0.0034422)
3: (0.0006841, 0.0087340, 0.0006841, 0.0087340, -0.0064275, 0.0064275)
4: (-0.0116961, -0.0046279, -0.0116961, -0.0046279, -0.0056436, 0.0056436)
5: (0.0093080, 0.0119852, 0.0093080, 0.0119852, -0.0021376, 0.0021376)
6: (0.0012563, 0.0114726, 0.0012563, 0.0114726, -0.0081573, 0.0081573)
7: (0.9789384, 0.9860873, 0.9789384, 0.9860873, -0.0057081, 0.0057081)
8: (-0.0091456, -0.0014809, -0.0091456, -0.0014809, -0.0061200, 0.0061200)
9: (-0.0040214, 0.0010416, -0.0040214, 0.0010416, -0.0040426, 0.0040426)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.62 + 2.34 = 3.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0036739, upper bound: 0.0036739

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 253
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 62
type: A, layer: 1, pos: 213

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036017, upper bound: 0.0035700
time: 1.45 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035703, upper bound: 0.0035704
time: 1.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.91 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 2.91
Output dim: 7, lower bound: -0.0036017, upper bound: 0.0035700
IS_A2, status: Status.VERIFIED, split count: 1, time: 2.91
Output dim: 7, lower bound: -0.0035703, upper bound: 0.0035704

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.96 + 2.91 = 6.87 seconds
