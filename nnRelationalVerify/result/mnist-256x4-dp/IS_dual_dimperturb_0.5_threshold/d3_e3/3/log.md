## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00071487


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0136726, -0.0098245, -0.0136726, -0.0098245, -0.0025120, 0.0025120)
1: (-0.0067935, -0.0057085, -0.0067935, -0.0057085, -0.0007082, 0.0007082)
2: (-0.0115639, -0.0035591, -0.0115639, -0.0035591, -0.0052255, 0.0052255)
3: (0.0000970, 0.0011563, 0.0000970, 0.0011563, -0.0006915, 0.0006915)
4: (0.0087517, 0.0147340, 0.0087517, 0.0147340, -0.0039052, 0.0039052)
5: (0.9979377, 0.9995998, 0.9979377, 0.9995998, -0.0010850, 0.0010850)
6: (0.0060117, 0.0075204, 0.0060117, 0.0075204, -0.0009848, 0.0009848)
7: (-0.0009468, 0.0046832, -0.0009468, 0.0046832, -0.0036752, 0.0036752)
8: (-0.0128378, -0.0084559, -0.0128378, -0.0084559, -0.0028604, 0.0028604)
9: (-0.0032802, -0.0029022, -0.0032802, -0.0029022, -0.0002468, 0.0002468)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.63 + 1.66 = 3.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0007325, upper bound: 0.0007326

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006900, upper bound: 0.0006815
time: 0.78 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0006900, upper bound: 0.0006900
time: 0.76 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.71 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 1.71
Output dim: 5, lower bound: -0.0006900, upper bound: 0.0006815
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.71
Output dim: 5, lower bound: -0.0006900, upper bound: 0.0006900

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.30 + 1.71 = 5.01 seconds
