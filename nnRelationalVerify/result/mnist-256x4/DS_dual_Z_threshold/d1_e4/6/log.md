## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.017292288


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0089681, 0.0080116, -0.0089681, 0.0080116, -0.0169797, 0.0169797)
1: (0.9901280, 1.0148224, 0.9901280, 1.0148224, -0.0246944, 0.0246944)
2: (-0.0118904, 0.0081498, -0.0118904, 0.0081498, -0.0199057, 0.0199058)
3: (-0.0007962, 0.0054384, -0.0007962, 0.0054384, -0.0062347, 0.0062347)
4: (-0.0099497, 0.0078145, -0.0099497, 0.0078145, -0.0177642, 0.0177642)
5: (-0.0033048, 0.0120104, -0.0033048, 0.0120104, -0.0153152, 0.0153152)
6: (-0.0137348, 0.0023293, -0.0137348, 0.0023293, -0.0160641, 0.0160641)
7: (-0.0107024, 0.0017518, -0.0107024, 0.0017518, -0.0124542, 0.0124542)
8: (-0.0158375, 0.0138741, -0.0158375, 0.0138741, -0.0295680, 0.0295680)
9: (-0.0087720, 0.0089540, -0.0087720, 0.0089540, -0.0177261, 0.0177261)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.77 + 3.74 = 5.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0180128, upper bound: 0.0180128

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 244
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171980, upper bound: 0.0171975
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0171975, upper bound: 0.0171982
time: 2.41 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.24 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 5.24
Output dim: 1, lower bound: -0.0171980, upper bound: 0.0171975
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 5.24
Output dim: 1, lower bound: -0.0171975, upper bound: 0.0171982

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 5.52 + 5.24 = 10.76 seconds
