## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.279429504


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0142819, 0.0515324, -0.0142819, 0.0515324, -0.0658142, 0.0658142)
1: (-0.0885537, 0.0610442, -0.0885537, 0.0610442, -0.1495979, 0.1495979)
2: (-0.0313161, 0.0638228, -0.0313161, 0.0638228, -0.0951389, 0.0951389)
3: (-0.0428029, 0.0392385, -0.0428029, 0.0392385, -0.0820414, 0.0820414)
4: (-0.0428602, 0.0839756, -0.0428602, 0.0839756, -0.1268357, 0.1268357)
5: (-0.0430394, 0.0690079, -0.0430394, 0.0690079, -0.1120473, 0.1120473)
6: (-0.0223391, 0.0418790, -0.0223391, 0.0418790, -0.0642181, 0.0642181)
7: (-0.0666547, 0.0700902, -0.0666547, 0.0700902, -0.1367449, 0.1367449)
8: (-0.0758634, 0.0571175, -0.0758634, 0.0571175, -0.1329810, 0.1329810)
9: (0.8029326, 1.1139959, 0.8029326, 1.1139959, -0.3110633, 0.3110633)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.96 + 3.72 = 5.68 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.2910724, upper bound: 0.2910724

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2790444, upper bound: 0.2790444
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2790444, upper bound: 0.2790444
time: 1.90 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 4.09 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 4.09
Output dim: 9, lower bound: -0.2790444, upper bound: 0.2790444
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 4.09
Output dim: 9, lower bound: -0.2790444, upper bound: 0.2790444

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 5.68 + 4.09 = 9.76 seconds
