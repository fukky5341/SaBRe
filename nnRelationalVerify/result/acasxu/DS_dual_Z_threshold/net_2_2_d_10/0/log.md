## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 2561.132074332682


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1091.8393555, 1704.6987305, -1091.8393555, 1704.6987305, -2796.5380859, 2796.5380859)
1: (-844.7684937, 1571.1506348, -844.7684937, 1571.1506348, -2415.9191895, 2415.9191895)
2: (-735.8467407, 1621.3187256, -735.8467407, 1621.3187256, -2357.1655273, 2357.1655273)
3: (-1145.6840820, 1614.1608887, -1145.6840820, 1614.1608887, -2759.8449707, 2759.8449707)
4: (-904.0911865, 1719.4404297, -904.0911865, 1719.4404297, -2623.5314941, 2623.5314941)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.43 + 2.11 = 3.54 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2561.2089106, upper bound: 2561.2089106

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2561.0892788, upper bound: 2561.0892788
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2561.0892788, upper bound: 2561.0892788
time: 0.70 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.54 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 1.54
Output dim: 0, lower bound: -2561.0892788, upper bound: 2561.0892788
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 1.54
Output dim: 0, lower bound: -2561.0892788, upper bound: 2561.0892788

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.54 + 1.54 = 5.08 seconds
