## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.07289891999999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720)
1: (-0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452)
2: (0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376)
3: (-0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086)
4: (0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.92 + 0.58 = 1.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0809988, upper bound: 0.0809988

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0782091, upper bound: 0.0784161
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0784161, upper bound: 0.0782091
time: 0.17 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.35 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.35
Output dim: 0, lower bound: -0.0782091, upper bound: 0.0784161
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.35
Output dim: 0, lower bound: -0.0784161, upper bound: 0.0782091

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0586231, upper bound: 0.0586231
time: 0.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0586231, upper bound: 0.0586231
time: 0.14 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720
1: -0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452
2: 0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376
3: -0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086
4: 0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0586231, upper bound: 0.0586231
time: 0.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0586231, upper bound: 0.0586231
time: 0.15 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.29 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 1.29
Output dim: 0, lower bound: -0.0586231, upper bound: 0.0586231
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 1.29
Output dim: 0, lower bound: -0.0586231, upper bound: 0.0586231
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 1.29
Output dim: 0, lower bound: -0.0586231, upper bound: 0.0586231
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 1.29
Output dim: 0, lower bound: -0.0586231, upper bound: 0.0586231

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.51 + 2.80 = 4.30 seconds
