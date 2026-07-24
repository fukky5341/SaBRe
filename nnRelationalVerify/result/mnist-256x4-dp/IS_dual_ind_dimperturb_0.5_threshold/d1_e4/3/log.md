## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000623675


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0018230, -0.0011028, -0.0018230, -0.0011028, -0.0004023, 0.0004023)
1: (-0.0089367, -0.0071091, -0.0089367, -0.0071091, -0.0010210, 0.0010210)
2: (0.0294857, 0.0306195, 0.0294857, 0.0306195, -0.0006334, 0.0006334)
3: (0.0023881, 0.0045053, 0.0023881, 0.0045053, -0.0011828, 0.0011828)
4: (-0.0079831, -0.0061242, -0.0079831, -0.0061242, -0.0010385, 0.0010385)
5: (0.0107144, 0.0114185, 0.0107144, 0.0114185, -0.0003934, 0.0003934)
6: (0.0034189, 0.0061059, 0.0034189, 0.0061059, -0.0015011, 0.0015011)
7: (0.9804517, 0.9823319, 0.9804517, 0.9823319, -0.0010504, 0.0010504)
8: (-0.0075231, -0.0055072, -0.0075231, -0.0055072, -0.0011262, 0.0011262)
9: (-0.0013618, -0.0000302, -0.0013618, -0.0000302, -0.0007439, 0.0007439)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.47 + 1.62 = 3.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0006392, upper bound: 0.0006392

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0006149, upper bound: 0.0006011
time: 0.78 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0006150, upper bound: 0.0006150
time: 0.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.65 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 1.65
Output dim: 7, lower bound: -0.0006149, upper bound: 0.0006011
IS_A2, status: Status.VERIFIED, split count: 1, time: 1.65
Output dim: 7, lower bound: -0.0006150, upper bound: 0.0006150

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.09 + 1.65 = 4.74 seconds
