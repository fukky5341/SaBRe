## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 3084.599462796909


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1569.5086670, 2773.5766602, -1569.5086670, 2773.5766602, -4343.0854492, 4343.0854492)
1: (-490.6097412, 1099.5386963, -490.6097412, 1099.5386963, -1590.1484375, 1590.1484375)
2: (-307.9147949, 1094.9610596, -307.9147949, 1094.9610596, -1402.8758545, 1402.8758545)
3: (-652.4367676, 1308.2292480, -652.4367676, 1308.2292480, -1960.6660156, 1960.6660156)
4: (-339.8120422, 1133.2332764, -339.8120422, 1133.2332764, -1473.0451660, 1473.0451660)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.75 + 2.23 = 2.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3084.6303091, upper bound: 3084.6303091

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3084.5432863, upper bound: 3084.5432863
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3084.5432863, upper bound: 3084.5432863
time: 0.83 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.68 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 1.68
Output dim: 0, lower bound: -3084.5432863, upper bound: 3084.5432863
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 1.68
Output dim: 0, lower bound: -3084.5432863, upper bound: 3084.5432863

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.98 + 1.68 = 4.66 seconds
