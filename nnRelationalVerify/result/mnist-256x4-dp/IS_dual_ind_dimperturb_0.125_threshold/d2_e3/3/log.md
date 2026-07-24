## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 7.432e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0081866, -0.0076237, -0.0081866, -0.0076237, -0.0002461, 0.0002461)
1: (-0.0052468, -0.0050881, -0.0052468, -0.0050881, -0.0000694, 0.0000694)
2: (-0.0001519, 0.0010190, -0.0001519, 0.0010190, -0.0005118, 0.0005118)
3: (0.0016072, 0.0017621, 0.0016072, 0.0017621, -0.0000677, 0.0000677)
4: (0.0053303, 0.0062054, 0.0053303, 0.0062054, -0.0003825, 0.0003825)
5: (0.9969872, 0.9972303, 0.9969872, 0.9972303, -0.0001063, 0.0001063)
6: (0.0051489, 0.0053696, 0.0051489, 0.0053696, -0.0000965, 0.0000965)
7: (-0.0041667, -0.0033432, -0.0041667, -0.0033432, -0.0003600, 0.0003600)
8: (-0.0065909, -0.0059499, -0.0065909, -0.0059499, -0.0002802, 0.0002802)
9: (-0.0034964, -0.0034411, -0.0034964, -0.0034411, -0.0000242, 0.0000242)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.45 + 1.26 = 2.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0000765, upper bound: 0.0000767

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 187

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000742, upper bound: 0.0000718
time: 0.43 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0000741, upper bound: 0.0000743
time: 0.43 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.04 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 1.04
Output dim: 5, lower bound: -0.0000742, upper bound: 0.0000718
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.04
Output dim: 5, lower bound: -0.0000741, upper bound: 0.0000743

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.71 + 1.04 = 3.75 seconds
