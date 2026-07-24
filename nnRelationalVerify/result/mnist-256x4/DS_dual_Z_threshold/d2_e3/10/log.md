## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.320591111


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.5426022, 0.4665188, -0.5426022, 0.4665188, -1.0091211, 1.0091211)
1: (-0.3630026, 0.3911184, -0.3630026, 0.3911184, -0.7541210, 0.7541210)
2: (-0.5269889, 0.4806983, -0.5269889, 0.4806983, -1.0076871, 1.0076871)
3: (0.7543352, 1.1590887, 0.7543352, 1.1590887, -0.4047535, 0.4047535)
4: (-0.4614211, 0.4625955, -0.4614211, 0.4625955, -0.9240167, 0.9240167)
5: (-0.4214845, 0.4672619, -0.4214845, 0.4672619, -0.8887464, 0.8887464)
6: (-0.5138452, 0.5371552, -0.5138452, 0.5371552, -1.0510004, 1.0510004)
7: (-0.4409947, 0.4028524, -0.4409947, 0.4028524, -0.8438470, 0.8438470)
8: (-0.4306176, 0.5533380, -0.4306176, 0.5533380, -0.9839556, 0.9839556)
9: (-0.4872051, 0.3854194, -0.4872051, 0.3854194, -0.8726246, 0.8726246)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.00 + 2.45 = 4.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.3305063, upper bound: 0.3305063

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 103
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2909420, upper bound: 0.2909420
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2909420, upper bound: 0.2909420
time: 1.17 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.61 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 2.61
Output dim: 3, lower bound: -0.2909420, upper bound: 0.2909420
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 2.61
Output dim: 3, lower bound: -0.2909420, upper bound: 0.2909420

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.45 + 2.61 = 7.06 seconds
