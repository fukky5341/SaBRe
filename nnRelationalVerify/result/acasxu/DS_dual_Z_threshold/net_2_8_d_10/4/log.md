## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 53.8414279856


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-208.9272308, 212.0743713, -208.9272308, 212.0743713, -421.0014954, 421.0014954)
1: (-20.8862324, 15.8918066, -20.8862324, 15.8918066, -36.7780342, 36.7780342)
2: (-34.6868706, 39.0282860, -34.6868706, 39.0282860, -73.7151566, 73.7151566)
3: (-40.3503723, 26.2559071, -40.3503723, 26.2559071, -66.6062698, 66.6062698)
4: (-30.3582592, 32.2099609, -30.3582592, 32.2099609, -62.5682144, 62.5682182)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.68 + 1.55 = 4.22 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -54.3304016, upper bound: 54.3304016

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8168306, upper bound: 53.8168306
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -53.8168306, upper bound: 53.8168306
time: 0.51 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.24 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 1.24
Output dim: 4, lower bound: -53.8168306, upper bound: 53.8168306
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 1.24
Output dim: 4, lower bound: -53.8168306, upper bound: 53.8168306

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.22 + 1.24 = 5.46 seconds
