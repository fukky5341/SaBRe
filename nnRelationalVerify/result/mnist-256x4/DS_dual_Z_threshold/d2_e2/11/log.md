## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.011909105


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0046572, 0.0033948, -0.0046572, 0.0033948, -0.0080520, 0.0080520)
1: (0.9879019, 1.0062624, 0.9879019, 1.0062624, -0.0183606, 0.0183606)
2: (-0.0157628, 0.0036852, -0.0157628, 0.0036852, -0.0189225, 0.0189225)
3: (0.0004106, 0.0063618, 0.0004106, 0.0063618, -0.0059512, 0.0059512)
4: (-0.0049218, 0.0108751, -0.0049218, 0.0108751, -0.0157968, 0.0157968)
5: (-0.0013414, 0.0120105, -0.0013414, 0.0120105, -0.0133519, 0.0133519)
6: (-0.0049703, 0.0050756, -0.0049703, 0.0050756, -0.0100459, 0.0100459)
7: (-0.0122603, -0.0013753, -0.0122603, -0.0013753, -0.0108851, 0.0108851)
8: (-0.0114670, 0.0189620, -0.0114670, 0.0189620, -0.0302684, 0.0302684)
9: (-0.0110455, 0.0063901, -0.0110455, 0.0063901, -0.0174356, 0.0174356)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.58 + 2.86 = 4.44 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0125356, upper bound: 0.0125359

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 193
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 234
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0118142, upper bound: 0.0118142
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0118142, upper bound: 0.0118142
time: 1.37 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.97 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 2.97
Output dim: 1, lower bound: -0.0118142, upper bound: 0.0118142
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 2.97
Output dim: 1, lower bound: -0.0118142, upper bound: 0.0118142

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.44 + 2.97 = 7.41 seconds
