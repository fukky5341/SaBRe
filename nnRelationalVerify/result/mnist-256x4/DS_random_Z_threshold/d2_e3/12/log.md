## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.314959394


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.7500529, 0.8076689, -0.7500529, 0.8076689, -1.5577219, 1.5577219)
1: (-0.3108026, 1.1302201, -0.3108026, 1.1302201, -1.4410226, 1.4410226)
2: (-0.5202457, 0.7794054, -0.5202457, 0.7794054, -1.2996511, 1.2996511)
3: (-0.4313014, 0.5804227, -0.4313014, 0.5804227, -1.0117240, 1.0117240)
4: (-0.6406322, 0.6517965, -0.6406322, 0.6517965, -1.2924287, 1.2924287)
5: (-0.6445287, 0.7135438, -0.6445287, 0.7135438, -1.3580725, 1.3580725)
6: (-0.5931337, 0.7638897, -0.5931337, 0.7638897, -1.3570234, 1.3570234)
7: (-0.5606133, 0.8349432, -0.5606133, 0.8349432, -1.3955564, 1.3955564)
8: (-0.7218040, 0.7847555, -0.7218040, 0.7847555, -1.5065595, 1.5065595)
9: (-0.6355414, 0.7659256, -0.6355414, 0.7659256, -1.4014671, 1.4014671)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.79 + 2.52 = 3.31 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -1.3417956, upper bound: 1.3417956

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 82
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 117

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2743508, upper bound: 1.2743508
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -1.2743508, upper bound: 1.2743508
time: 1.08 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.18 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 2.18
Output dim: 1, lower bound: -1.2743508, upper bound: 1.2743508
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 2.18
Output dim: 1, lower bound: -1.2743508, upper bound: 1.2743508

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.31 + 2.18 = 5.49 seconds
