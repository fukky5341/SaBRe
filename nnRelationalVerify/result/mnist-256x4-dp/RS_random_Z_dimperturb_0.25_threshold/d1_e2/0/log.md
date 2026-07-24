## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000280174


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0009866, 0.0011007, 0.0009866, 0.0011007, -0.0001105, 0.0001105)
1: (0.9936137, 0.9939742, 0.9936137, 0.9939742, -0.0003142, 0.0003142)
2: (-0.0066230, -0.0052000, -0.0066230, -0.0052000, -0.0011641, 0.0011641)
3: (0.0038432, 0.0040182, 0.0038432, 0.0040182, -0.0001439, 0.0001439)
4: (0.0025269, 0.0036515, 0.0025269, 0.0036515, -0.0009730, 0.0009730)
5: (0.0061088, 0.0064723, 0.0061088, 0.0064723, -0.0003636, 0.0003636)
6: (-0.0014200, -0.0009261, -0.0014200, -0.0009261, -0.0004011, 0.0004011)
7: (-0.0082882, -0.0080108, -0.0082882, -0.0080108, -0.0002775, 0.0002775)
8: (0.0050839, 0.0069535, 0.0050839, 0.0069535, -0.0013897, 0.0013897)
9: (-0.0036858, -0.0033387, -0.0036858, -0.0033387, -0.0003471, 0.0003471)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.19 + 1.40 = 2.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0002938, upper bound: 0.0002938

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 127
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 64

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002380, upper bound: 0.0002380
time: 0.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0002380, upper bound: 0.0002380
time: 0.51 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.03 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.03
Output dim: 1, lower bound: -0.0002380, upper bound: 0.0002380
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.03
Output dim: 1, lower bound: -0.0002380, upper bound: 0.0002380

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 2.59 + 1.03 = 3.62 seconds
