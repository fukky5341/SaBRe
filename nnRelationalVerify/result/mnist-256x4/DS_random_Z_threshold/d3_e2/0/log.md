## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.7875191039999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.2635888, 0.2611926, -0.2635888, 0.2611926, -0.5247813, 0.5247813)
1: (-0.1908717, 0.2009175, -0.1908717, 0.2009175, -0.3917893, 0.3917893)
2: (-0.1701133, 0.2907368, -0.1701133, 0.2907368, -0.4608501, 0.4608501)
3: (0.1429092, 1.0597293, 0.1429092, 1.0597293, -0.9168201, 0.9168201)
4: (-0.2238975, 0.2077588, -0.2238975, 0.2077588, -0.4316563, 0.4316563)
5: (-0.1434896, 0.6777036, -0.1434896, 0.6777036, -0.8211932, 0.8211932)
6: (-0.2152857, 0.2608757, -0.2152857, 0.2608757, -0.4761614, 0.4761614)
7: (-0.3123749, 0.2203892, -0.3123749, 0.2203892, -0.5327642, 0.5327642)
8: (-0.2114335, 0.2966909, -0.2114335, 0.2966909, -0.5081244, 0.5081244)
9: (-0.3438668, 0.3164717, -0.3438668, 0.3164717, -0.6603385, 0.6603385)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.85 + 2.70 = 3.55 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.8203325, upper bound: 0.8203325

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7534933, upper bound: 0.7534933
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7534933, upper bound: 0.7534933
time: 1.11 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.22 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 2.22
Output dim: 3, lower bound: -0.7534933, upper bound: 0.7534933
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 2.22
Output dim: 3, lower bound: -0.7534933, upper bound: 0.7534933

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.55 + 2.22 = 5.78 seconds
