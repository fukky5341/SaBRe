## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.851544332


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.3877775, 0.4621432, -0.3877775, 0.4621432, -0.8499205, 0.8499205)
1: (-0.5284067, 2.0988278, -0.5284067, 2.0988278, -2.6272345, 2.6272345)
2: (-0.3809817, 0.5539122, -0.3809817, 0.5539122, -0.9348937, 0.9348938)
3: (-0.3085160, 0.3773541, -0.3085160, 0.3773541, -0.6858702, 0.6858701)
4: (-0.3952212, 0.4911242, -0.3952212, 0.4911242, -0.8863454, 0.8863454)
5: (-0.4354007, 0.5080089, -0.4354007, 0.5080089, -0.9434096, 0.9434096)
6: (-0.4014111, 0.4879104, -0.4014111, 0.4879104, -0.8893216, 0.8893216)
7: (-0.3190002, 0.8904487, -0.3190002, 0.8904487, -1.2094488, 1.2094488)
8: (-0.2782317, 0.7783343, -0.2782317, 0.7783343, -1.0565660, 1.0565660)
9: (-0.4433337, 0.5317172, -0.4433337, 0.5317172, -0.9750509, 0.9750510)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.03 + 6.08 = 8.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -1.8702468, upper bound: 1.8702468

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 151
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8426265, upper bound: 1.8426265
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.8426265, upper bound: 1.8426265
time: 3.82 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.70 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 7.70
Output dim: 1, lower bound: -1.8426265, upper bound: 1.8426265
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 7.70
Output dim: 1, lower bound: -1.8426265, upper bound: 1.8426265

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 8.11 + 7.70 = 15.80 seconds
