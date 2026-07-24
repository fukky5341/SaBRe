## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 2.776e-05


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0001891, 0.0006036, 0.0001891, 0.0006036, -0.0002194, 0.0002194)
1: (-0.0034473, -0.0033746, -0.0034473, -0.0033746, -0.0000227, 0.0000227)
2: (0.0151729, 0.0156948, 0.0151729, 0.0156948, -0.0002599, 0.0002599)
3: (1.0067765, 1.0068991, 1.0067765, 1.0068991, -0.0000925, 0.0000925)
4: (-0.0042035, -0.0041236, -0.0042035, -0.0041236, -0.0000367, 0.0000367)
5: (0.0041241, 0.0044406, 0.0041241, 0.0044406, -0.0001662, 0.0001662)
6: (-0.0025955, -0.0025750, -0.0025955, -0.0025750, -0.0000172, 0.0000172)
7: (-0.0125408, -0.0117708, -0.0125408, -0.0117708, -0.0004986, 0.0004986)
8: (-0.0132188, -0.0123906, -0.0132188, -0.0123906, -0.0003595, 0.0003595)
9: (0.0020150, 0.0024038, 0.0020150, 0.0024038, -0.0001562, 0.0001562)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.52 + 1.24 = 2.76 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0000299, upper bound: 0.0000301

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3
type: RSZ, layer: 3, pos: 107
type: RSZ, layer: 3, pos: 189

Time for candidate selection: 0.27 seconds

### Candidate
type: RSZ, layer: 3, pos: 3

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000275, upper bound: 0.0000243
time: 0.42 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0000241, upper bound: 0.0000276
time: 0.41 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.11 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.11
Output dim: 3, lower bound: -0.0000275, upper bound: 0.0000243
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.11
Output dim: 3, lower bound: -0.0000241, upper bound: 0.0000276

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.76 + 1.11 = 3.87 seconds
